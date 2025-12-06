"""
Bayesian Network for Predicting Soviet Literary Journal Publications
Based on CS221 Lecture 12: Bayesian Networks structure
"""

import numpy as np
from einops import einsum
from collections import defaultdict
from typing import Callable, Dict, List, Tuple
import pandas as pd


class ProbTable:
    """
    Represents an arbitrary probability table: could be a local conditional
    distribution, a marginal distribution, or a conditional distribution.
    """
    def __init__(self, description: str, data: np.ndarray | Callable, shape: tuple = None):
        if callable(data):
            # Build up the probabilities
            self.probs = np.empty(shape)
            def recurse(assignment: list):
                if len(assignment) == len(shape):
                    self.probs[tuple(assignment)] = data(*assignment)
                else:
                    for i in range(shape[len(assignment)]):
                        recurse(assignment + [i])
            recurse([])
        else:
            self.probs = np.array(data)
        
        # Parse the description
        self.cond_vars = []
        self.gen_vars = []
        self.cond_vals = []
        self.gen_vals = []
        items = description.split(" ")
        on_conditioning_side = False
        for item in items:
            if item == "|":
                on_conditioning_side = True
            elif on_conditioning_side:
                if "=" in item:
                    self.cond_vals.append(item)
                else:
                    self.cond_vars.append(item)
            else:
                if "=" in item:
                    self.gen_vals.append(item)
                else:
                    self.gen_vars.append(item)
    
    @property
    def p(self):
        return self.probs


class SovietJournalBayesianNetwork:
    """
    Bayesian Network Structure for Soviet Literary Journal Publications:
    
    Decade (D) → General Secretary (GS)
    General Secretary (GS) → Published (P)
    Decade (D) → Published (P)
    Author (A) → Published (P)
    
    If published, then:
    General Secretary (GS) → Journal (J)
    Decade (D) → Journal (J)
    Author (A) → Journal (J)
    
    Goal: Predict which American authors would be published and in which journal
    """
    
    def __init__(self, learn_from_data=False, data_path=None):
        # Define categorical mappings
        self.decades = ['1950s', '1960s', '1970s', '1980s', '1990s']
        self.general_secretaries = ['Stalin', 'Khrushchev', 'Brezhnev', 'Andropov', 'Chernenko', 'Gorbachev']
        self.journals = ['IL', 'Vsesvit']  # IL = Inostrannaia Literatura
        
        # Authors will be learned from data
        self.authors = []
        self.author_map = {}
        
        # For learned mappings
        self.decade_map = {}
        self.gs_map = {}
        self.journal_map = {}
        self.editor_map = {}
        
        self.learn_from_data = learn_from_data
        self.data = None
        
        if learn_from_data and data_path:
            self._load_and_preprocess_data(data_path)
            self._learn_from_data()
        else:
            # Initialize network with prior knowledge
            self._build_network()
    
    def _load_and_preprocess_data(self, data_path: str):
        """
        Load and preprocess the CSV data
        """
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        
        print(f"Loaded {len(self.data)} records")
        print(f"Columns: {list(self.data.columns)}")
        
        # Check for english_author column instead of author
        if 'english_author' not in self.data.columns:
            raise ValueError("CSV must have an 'english_author' column!")
        
        # Create mappings from raw data to indices
        self._create_mappings()
        
        # Add encoded columns - ENSURE THEY ARE INTEGERS
        self.data['decade_idx'] = self.data['decade'].map(self.decade_map).fillna(-1).astype(int)
        self.data['gs_idx'] = self.data['general_secretary'].map(self.gs_map).fillna(-1).astype(int)
        self.data['journal_idx'] = self.data['journal'].map(self.journal_map).fillna(-1).astype(int)
        self.data['author_idx'] = self.data['english_author'].map(self.author_map).fillna(-1).astype(int)
        
        # Handle editor if exists
        if 'editor_in_chief' in self.data.columns:
            editors = sorted(self.data['editor_in_chief'].dropna().unique())
            self.editor_map = {editor: idx for idx, editor in enumerate(editors)}
            self.editors = editors
            self.data['editor_idx'] = self.data['editor_in_chief'].map(self.editor_map).fillna(-1).astype(int)
        
        # Remove rows with missing critical data
        valid_mask = (
            (self.data['decade_idx'] >= 0) & 
            (self.data['gs_idx'] >= 0) & 
            (self.data['journal_idx'] >= 0) &
            (self.data['author_idx'] >= 0)
        )
        
        print(f"Removing {(~valid_mask).sum()} rows with missing critical data")
        self.data = self.data[valid_mask].copy()
        
        print(f"Final dataset: {len(self.data)} records")
        print(f"Unique authors: {len(self.authors)}")
        print(f"Journal distribution: IL={sum(self.data['journal_idx']==0)}, Vsesvit={sum(self.data['journal_idx']==1)}")
        print()
    
    def _create_mappings(self):
        """
        Create mappings from raw categorical values to indices
        """
        # Author mapping - learn from english_author column
        authors = sorted(self.data['english_author'].dropna().unique())
        self.authors = authors
        self.author_map = {author: idx for idx, author in enumerate(authors)}
        
        # Decade mapping
        for idx, decade in enumerate(self.decades):
            self.decade_map[decade] = idx
            # Also handle year format
            year_start = 1950 + idx * 10
            for year in range(year_start, year_start + 10):
                self.decade_map[str(year)] = idx
        
        # General Secretary mapping
        for idx, gs in enumerate(self.general_secretaries):
            self.gs_map[gs] = idx
        
        # Journal mapping
        self.journal_map = {'IL': 0, 'Inostrannaia Literatura': 0, 'Vsesvit': 1}
    
    def _learn_from_data(self):
        """
        Learn local conditional probabilities from data using MLE
        """
        print("Learning conditional probabilities from data...")
        print("=" * 80)
        
        data = self.data
        num_authors = len(self.authors)
        
        # Learn p(D) - decade marginal
        decade_counts = data['decade_idx'].value_counts().sort_index()
        decade_probs = np.zeros(5)
        for idx, count in decade_counts.items():
            decade_probs[int(idx)] = count
        decade_probs = decade_probs / decade_probs.sum()
        self.p_decade = ProbTable("D", decade_probs)
        
        print("Learned p(Decade):")
        for i, decade in enumerate(self.decades):
            print(f"  {decade:8} = {decade_probs[i]:.3f}")
        print()
        
        # Learn p(GS|D) - general secretary given decade
        gs_given_d = np.zeros((5, 6))  # [decade, gs]
        for d_idx in range(5):
            decade_data = data[data['decade_idx'] == d_idx]
            if len(decade_data) > 0:
                gs_counts = decade_data['gs_idx'].value_counts()
                for gs_idx, count in gs_counts.items():
                    gs_given_d[d_idx, int(gs_idx)] = count
                # Normalize + Laplace smoothing
                gs_given_d[d_idx] += 0.1
                gs_given_d[d_idx] /= gs_given_d[d_idx].sum()
            else:
                gs_given_d[d_idx] = np.ones(6) / 6  # Uniform if no data
        
        self.p_gs_given_d = ProbTable("GS | D", gs_given_d)
        
        print("Learned p(GS | Decade):")
        print("       ", " ".join([f"{gs[:4]:>6}" for gs in self.general_secretaries]))
        for i, decade in enumerate(self.decades):
            probs_str = " ".join([f"{gs_given_d[i,j]:6.3f}" for j in range(6)])
            print(f"  {decade:8} {probs_str}")
        print()
        
        # Learn p(A|D,GS) - author "popularity" given decade and general secretary
        # This captures which authors were favored in different eras
        author_given_d_gs = np.zeros((5, 6, num_authors))  # [decade, gs, author]
        for d_idx in range(5):
            for gs_idx in range(6):
                subset = data[(data['decade_idx'] == d_idx) & (data['gs_idx'] == gs_idx)]
                if len(subset) > 0:
                    author_counts = subset['author_idx'].value_counts()
                    for a_idx, count in author_counts.items():
                        author_given_d_gs[d_idx, gs_idx, int(a_idx)] = count
                    # Normalize + Laplace smoothing
                    author_given_d_gs[d_idx, gs_idx] += 0.01
                    author_given_d_gs[d_idx, gs_idx] /= author_given_d_gs[d_idx, gs_idx].sum()
                else:
                    # Uniform if no data for this combo
                    author_given_d_gs[d_idx, gs_idx] = np.ones(num_authors) / num_authors
        
        self.p_author_given_d_gs = ProbTable("A | D GS", author_given_d_gs)
        
        # Show top authors by era
        print("Top 5 most published authors by General Secretary:")
        for gs_idx, gs in enumerate(self.general_secretaries):
            # Marginalize over decades, weighted by p(D|GS)
            author_probs = np.zeros(num_authors)
            for d_idx in range(5):
                if gs_given_d[d_idx, gs_idx] > 0.01:  # If this GS was active in this decade
                    author_probs += author_given_d_gs[d_idx, gs_idx] * decade_probs[d_idx]
            
            if author_probs.sum() > 0:
                author_probs /= author_probs.sum()
                top_indices = np.argsort(author_probs)[-5:][::-1]
                print(f"  {gs:15}:", ", ".join([self.authors[i] for i in top_indices if author_probs[i] > 0.01]))
        print()
        
        # Learn p(J|A,D,GS) - journal given author, decade, and general secretary
        journal_given_a_d_gs = np.zeros((num_authors, 5, 6, 2))  # [author, decade, gs, journal]
        
        for a_idx in range(num_authors):
            for d_idx in range(5):
                for gs_idx in range(6):
                    subset = data[(data['author_idx'] == a_idx) & 
                                  (data['decade_idx'] == d_idx) & 
                                  (data['gs_idx'] == gs_idx)]
                    if len(subset) > 0:
                        journal_counts = subset['journal_idx'].value_counts()
                        for j_idx, count in journal_counts.items():
                            journal_given_a_d_gs[a_idx, d_idx, gs_idx, int(j_idx)] = count
                        # Normalize + Laplace smoothing
                        journal_given_a_d_gs[a_idx, d_idx, gs_idx] += 0.5
                        journal_given_a_d_gs[a_idx, d_idx, gs_idx] /= journal_given_a_d_gs[a_idx, d_idx, gs_idx].sum()
                    else:
                        # Use author-only or overall marginal
                        author_subset = data[data['author_idx'] == a_idx]
                        if len(author_subset) > 0:
                            journal_counts = author_subset['journal_idx'].value_counts()
                            for j_idx, count in journal_counts.items():
                                journal_given_a_d_gs[a_idx, d_idx, gs_idx, int(j_idx)] = count
                            journal_given_a_d_gs[a_idx, d_idx, gs_idx] += 0.5
                            journal_given_a_d_gs[a_idx, d_idx, gs_idx] /= journal_given_a_d_gs[a_idx, d_idx, gs_idx].sum()
                        else:
                            journal_given_a_d_gs[a_idx, d_idx, gs_idx] = [0.528, 0.472]  # Overall marginal
        
        self.p_journal_given_a_d_gs = ProbTable("J | A D GS", journal_given_a_d_gs)
        
        print("Sample: Journal preferences for top authors (marginalized over time):")
        author_counts = data['author_idx'].value_counts()
        top_authors = author_counts.head(10).index
        for a_idx in top_authors:
            a_idx = int(a_idx)
            author_name = self.authors[a_idx]
            # Compute overall journal preference
            author_data = data[data['author_idx'] == a_idx]
            il_count = (author_data['journal_idx'] == 0).sum()
            vs_count = (author_data['journal_idx'] == 1).sum()
            total = il_count + vs_count
            print(f"  {author_name:25} → IL: {il_count:2d}/{total:2d} ({il_count/total:.2f})")
        print()
        
        # Compute joint distribution
        self._compute_joint_distribution()
        
        print("=" * 80)
        print("Learning complete!")
        print("=" * 80)
        print()
    
    def _build_network(self):
        """
        Step 1-3: Define variables, edges, and local conditional probabilities
        Using prior knowledge (not learned from data)
        """
        
        # Decade probabilities
        decade_probs = [0.05, 0.08, 0.35, 0.47, 0.05]  # More publications in 70s-80s
        self.p_decade = ProbTable("D", decade_probs)
        
        # General Secretary given Decade
        # GS progression: Stalin(40s-53), Khrushchev(53-64), Brezhnev(64-82), 
        # Andropov(82-84), Chernenko(84-85), Gorbachev(85-91)
        def p_gs_given_decade(d, gs):
            # d: decade index, gs: general secretary index
            if d == 0:  # 1950s
                return 0.5 if gs == 1 else 0.02  # Mostly Khrushchev
            elif d == 1:  # 1960s
                return 0.7 if gs == 2 else 0.05  # Mostly Brezhnev
            elif d == 2:  # 1970s
                return 0.95 if gs == 2 else 0.01  # Almost all Brezhnev
            elif d == 3:  # 1980s
                probs = [0.0, 0.0, 0.50, 0.15, 0.05, 0.30]  # Brezhnev, Andropov, Chernenko, Gorbachev
                return probs[gs]
            else:  # 1990s
                return 0.95 if gs == 5 else 0.01  # Almost all Gorbachev
        
        self.p_gs_given_d = ProbTable("GS | D", p_gs_given_decade, shape=(5, 6))
        
        # Journal given Decade and General Secretary
        def p_journal_given_features(d, gs, j):
            """
            Key insights from the data:
            - IL becomes dominant in later decades
            - Vsesvit stronger in early decades
            - Gorbachev era increases IL dominance
            """
            base_prob = 0.528  # IL base rate
            
            # Decade effects
            if d == 0:  # 1950s
                base_prob -= 0.10  # Vsesvit stronger
            elif d == 1:  # 1960s
                base_prob -= 0.05
            elif d >= 3:  # 1980s-1990s
                base_prob += 0.10  # IL dominant
            
            # General Secretary effects
            if gs == 5:  # Gorbachev - more open, IL dominant
                base_prob += 0.08
            elif gs == 1:  # Khrushchev - Vsesvit stronger
                base_prob -= 0.08
            
            # Ensure valid probability
            base_prob = max(0.01, min(0.99, base_prob))
            
            if j == 0:  # IL
                return base_prob
            else:  # Vsesvit
                return 1 - base_prob
        
        self.p_journal_given_d_gs = ProbTable("J | D GS", p_journal_given_features, shape=(5, 6, 2))
        
        # Published given Journal, Decade, General Secretary
        def p_published_given_features(j, d, gs, p):
            """
            Publishability depends on:
            - Political climate (GS)
            - Journal capacity
            - Decade (cultural openness)
            """
            base_pub_prob = 0.3  # Base publication probability
            
            # Decade effects (cultural policies)
            if d >= 3:  # 1980s-1990s - more open
                base_pub_prob += 0.20
            elif d == 2:  # 1970s - moderate
                base_pub_prob += 0.10
            
            # General Secretary effects (political openness)
            if gs == 5:  # Gorbachev - glasnost, most permissive
                base_pub_prob += 0.25
            elif gs == 1:  # Khrushchev - thaw
                base_pub_prob += 0.10
            elif gs == 2:  # Brezhnev - stable but controlled
                base_pub_prob += 0.05
            elif gs == 3 or gs == 4:  # Andropov/Chernenko - restrictive
                base_pub_prob -= 0.05
            
            # Ensure valid probability
            base_pub_prob = max(0.05, min(0.95, base_pub_prob))
            
            if p == 1:  # published
                return base_pub_prob
            else:  # not published
                return 1 - base_pub_prob
        
        self.p_published_given_j_d_gs = ProbTable("P | J D GS", p_published_given_features, shape=(2, 5, 6, 2))
        
        # Step 4: Define joint distribution
        self._compute_joint_distribution()
    
    def _compute_joint_distribution(self):
        """
        Step 4: Joint distribution as product of local conditional probabilities
        P(D, GS, A, J) = p(D) * p(GS|D) * p(A|D,GS) * p(J|A,D,GS)
        
        All observed data is published, so we compute P(D,GS,A,J | Published=1)
        """
        # First compute P(D, GS) = p(D) * p(GS|D)
        P_D_GS = ProbTable("D GS", einsum(self.p_decade.p, self.p_gs_given_d.p, "d, d gs -> d gs"))
        
        # Add Author: P(D, GS, A) = P(D, GS) * p(A|D,GS)
        P_D_GS_A = ProbTable("D GS A", einsum(
            P_D_GS.p, self.p_author_given_d_gs.p, "d gs, d gs a -> d gs a"
        ))
        
        # Add Journal: P(D, GS, A, J) = P(D, GS, A) * p(J|A,D,GS)
        self.P_full = ProbTable("D GS A J", einsum(
            P_D_GS_A.p, self.p_journal_given_a_d_gs.p, "d gs a, a d gs j -> d gs a j"
        ))
    
    def query(self, query_vars: str, evidence: Dict[str, int]) -> ProbTable:
        """
        Probabilistic Inference: P(query | evidence)
        
        Args:
            query_vars: e.g., "J" for journal, "A" for author
            evidence: dict like {"D": 2, "GS": 2, "A": 5}
        
        Returns:
            ProbTable with conditional distribution
        """
        # Start with full joint distribution
        current = self.P_full.p.copy()
        
        # Variable order in joint distribution: D, GS, A, J
        var_order = ['D', 'GS', 'A', 'J']
        
        # Apply evidence by selecting specific values
        for var, val in evidence.items():
            if var in var_order:
                var_idx = var_order.index(var)
                # Select only the specified value along this axis
                current = np.take(current, val, axis=var_idx)
                # Update var_order to reflect removed dimension
                var_order.remove(var)
        
        # Marginalize out non-query variables
        if query_vars in var_order:
            query_idx = var_order.index(query_vars)
            # Sum over all axes except query axis
            axes_to_sum = tuple(i for i in range(len(var_order)) if i != query_idx)
            if axes_to_sum:
                marginal = np.sum(current, axis=axes_to_sum)
            else:
                marginal = current
        else:
            # Query variable already in evidence, return scalar
            marginal = np.sum(current)
        
        # Normalize
        prob_evidence = np.sum(marginal)
        if prob_evidence > 0:
            conditional = marginal / prob_evidence
        else:
            conditional = marginal
        
        return ProbTable(f"{query_vars} | evidence", conditional)
    
    def predict_author_publishability(self, author_name: str, decade_idx: int, gs_idx: int) -> Tuple[float, str, float]:
        """
        Predict if an author would be published and in which journal
        
        Args:
            author_name: Name of the author
            decade_idx: 0-4 (1950s-1990s)
            gs_idx: 0-5 (Stalin-Gorbachev)
        
        Returns:
            (publishability_score, predicted_journal, journal_confidence)
        """
        if author_name not in self.author_map:
            # Unknown author - use base rates
            evidence = {"D": decade_idx, "GS": gs_idx}
            
            # Get distribution over all authors for this era
            result_authors = self.query("A", evidence)
            # Use entropy as uncertainty measure - lower entropy = more predictable era
            author_probs = result_authors.p
            entropy = -np.sum(author_probs * np.log(author_probs + 1e-10))
            max_entropy = np.log(len(self.authors))
            
            # Publishability is lower for unknown authors, scaled by era openness
            base_rate = 0.3
            if gs_idx == 5:  # Gorbachev
                base_rate = 0.5
            elif gs_idx == 1:  # Khrushchev
                base_rate = 0.4
            
            publishability = base_rate * (1 - entropy / max_entropy)
            
            # Predict journal using base rates
            result_journal = self.query("J", evidence)
            journal_probs = result_journal.p
            pred_journal_idx = np.argmax(journal_probs)
            journal_conf = journal_probs[pred_journal_idx]
            
            return publishability, self.journals[pred_journal_idx], journal_conf
        
        # Known author
        author_idx = self.author_map[author_name]
        evidence = {"D": decade_idx, "GS": gs_idx, "A": author_idx}
        
        # Get publication probability from conditional distribution
        # Since all our data is published, we estimate based on frequency in this era
        author_probs_in_era = self.p_author_given_d_gs.p[decade_idx, gs_idx]
        publishability = author_probs_in_era[author_idx]
        
        # Normalize to [0, 1] range
        max_prob = np.max(author_probs_in_era)
        if max_prob > 0:
            publishability = publishability / max_prob
        
        # Predict journal
        result_journal = self.query("J", evidence)
        journal_probs = result_journal.p
        pred_journal_idx = np.argmax(journal_probs)
        journal_conf = journal_probs[pred_journal_idx]
        
        return publishability, self.journals[pred_journal_idx], journal_conf
    
    def predict_journal_for_author(self, author_name: str, decade_idx: int, gs_idx: int) -> Tuple[str, float]:
        """
        Predict which journal would publish a specific author
        
        Args:
            author_name: Name of the author
            decade_idx: 0-4 (1950s-1990s)
            gs_idx: 0-5 (Stalin-Gorbachev)
        
        Returns:
            (journal_name, confidence)
        """
        if author_name not in self.author_map:
            print(f"Warning: Author '{author_name}' not in training data")
            # Use base rates
            evidence = {"D": decade_idx, "GS": gs_idx}
        else:
            author_idx = self.author_map[author_name]
            evidence = {"D": decade_idx, "GS": gs_idx, "A": author_idx}
        
        result = self.query("J", evidence)
        journal_probs = result.p
        pred_journal_idx = np.argmax(journal_probs)
        confidence = journal_probs[pred_journal_idx]
        
        return self.journals[pred_journal_idx], confidence
    
    def get_top_authors_for_era(self, decade_idx: int, gs_idx: int, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most likely authors to be published in a given era
        
        Args:
            decade_idx: 0-4
            gs_idx: 0-5
            top_k: Number of top authors to return
        
        Returns:
            List of (author_name, probability) tuples
        """
        evidence = {"D": decade_idx, "GS": gs_idx}
        result = self.query("A", evidence)
        
        author_probs = result.p
        top_indices = np.argsort(author_probs)[-top_k:][::-1]
        
        top_authors = [(self.authors[idx], author_probs[idx]) for idx in top_indices]
        return top_authors
    
    def evaluate_on_data(self, test_data: pd.DataFrame = None) -> Dict:
        """
        Evaluate model performance on test data
        
        Returns:
            Dictionary with accuracy, precision, recall, F1
        """
        if test_data is None:
            if self.data is None:
                print("No data available for evaluation")
                return {}
            # Use 20% of data as test set
            test_size = int(0.2 * len(self.data))
            test_data = self.data.sample(n=test_size, random_state=42)
        
        print("Evaluating model on test data...")
        print(f"Test set size: {len(test_data)}")
        
        # Evaluation: Journal prediction for known authors
        predictions = []
        actuals = []
        confidences = []
        
        for _, row in test_data.iterrows():
            # FIXED: Use english_author from data
            author_name = row['english_author']
            decade_idx = int(row['decade_idx'])
            gs_idx = int(row['gs_idx'])
            actual_journal = int(row['journal_idx'])
            
            pred_journal, conf = self.predict_journal_for_author(author_name, decade_idx, gs_idx)
            pred_journal_idx = self.journals.index(pred_journal)
            
            predictions.append(pred_journal_idx)
            actuals.append(actual_journal)
            confidences.append(conf)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        accuracy = (predictions == actuals).mean()
        
        # Precision, Recall, F1 for IL (class 0)
        tp = ((predictions == 0) & (actuals == 0)).sum()
        fp = ((predictions == 0) & (actuals == 1)).sum()
        fn = ((predictions == 1) & (actuals == 0)).sum()
        tn = ((predictions == 1) & (actuals == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'precision_IL': precision,
            'recall_IL': recall,
            'f1_IL': f1,
            'mean_confidence': np.mean(confidences),
            'confusion_matrix': {
                'TP': int(tp), 'FP': int(fp), 
                'FN': int(fn), 'TN': int(tn)
            }
        }
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS - Journal Prediction for Authors")
        print("=" * 80)
        print(f"Test Accuracy:        {accuracy:.3f}")
        print(f"Precision (IL):       {precision:.3f}")
        print(f"Recall (IL):          {recall:.3f}")
        print(f"F1 Score (IL):        {f1:.3f}")
        print(f"Mean Confidence:      {np.mean(confidences):.3f}")
        print(f"\nConfusion Matrix:")
        print(f"                Predicted IL    Predicted Vsesvit")
        print(f"Actual IL            {tp:4d}              {fn:4d}")
        print(f"Actual Vsesvit       {fp:4d}              {tn:4d}")
        print("=" * 80)
        print()
        
        return results
    
    def analyze_explaining_away(self):
        """
        Demonstrate the explaining away phenomenon in the network
        """
        print("=" * 80)
        print("EXPLAINING AWAY ANALYSIS")
        print("=" * 80)
        print("\nScenario: What if we know an author was published in the 1980s?")
        print("Question: Does knowing which journal change our belief about who they might be?")
        print()
        
        # Get top authors for 1980s
        top_authors_1980s = self.get_top_authors_for_era(decade_idx=3, gs_idx=2, top_k=5)  # Brezhnev era
        
        print("Top 5 authors in 1980s (Brezhnev):")
        for author, prob in top_authors_1980s:
            print(f"  {author:25} = {prob:.4f}")
        print()
        
        # Now condition on journal = IL
        print("If we know it was published in IL:")
        evidence_il = {"D": 3, "GS": 2, "J": 0}  # 1980s, Brezhnev, IL
        result_il = self.query("A", evidence_il)
        top_indices_il = np.argsort(result_il.p)[-5:][::-1]
        for idx in top_indices_il:
            print(f"  {self.authors[idx]:25} = {result_il.p[idx]:.4f}")
        print()
        
        # Now condition on journal = Vsesvit
        print("If we know it was published in Vsesvit:")
        evidence_vs = {"D": 3, "GS": 2, "J": 1}  # 1980s, Brezhnev, Vsesvit
        result_vs = self.query("A", evidence_vs)
        top_indices_vs = np.argsort(result_vs.p)[-5:][::-1]
        for idx in top_indices_vs:
            print(f"  {self.authors[idx]:25} = {result_vs.p[idx]:.4f}")
        print()
        
        print("Interpretation:")
        print("  - Knowing the journal 'explains away' uncertainty about the author")
        print("  - Different journals had different author preferences")
        print("  - This is the explaining away phenomenon from lecture!")
        print("=" * 80)
        print()


def rejection_sampling(program: Callable, query: Callable, evidence: Callable, num_samples: int) -> Dict:
    """
    Rejection sampling for approximate inference
    
    Args:
        program: Function that generates samples from joint distribution
        query: Function that extracts query variable from sample
        evidence: Function that checks if sample matches evidence
        num_samples: Number of samples to draw
    
    Returns:
        Dictionary of probabilities for each query value
    """
    counts = defaultdict(int)
    
    for _ in range(num_samples):
        sample = program()
        if evidence(sample):
            counts[query(sample)] += 1
    
    # Normalize
    total = sum(counts.values())
    if total == 0:
        return {}
    
    probs = {q: counts[q] / total for q in counts}
    return probs


def create_example_csv(output_path: str = "example_soviet_journals.csv"):
    """
    Create an example CSV file showing the expected format
    This demonstrates what your data should look like
    """
    example_data = {
        'journal': ['IL', 'Vsesvit', 'IL', 'Vsesvit', 'IL', 'Vsesvit', 'IL', 'Vsesvit'],
        'english_author': ['Ray Bradbury', 'Langston Hughes', 'Ernest Hemingway', 'William Faulkner', 
                   'Isaac Asimov', 'John Steinbeck', 'Philip K. Dick', 'Tennessee Williams'],
        'title': ['The Martian Chronicles', 'Selected Poems', 'The Old Man and the Sea', 
                  'The Sound and the Fury', 'I, Robot', 'The Grapes of Wrath', 
                  'Do Androids Dream', 'A Streetcar Named Desire'],
        'year': [1975, 1967, 1977, 1978, 1982, 1968, 1985, 1989],
        'decade': ['1970s', '1960s', '1970s', '1970s', '1980s', '1960s', '1980s', '1980s'],
        'general_secretary': ['Brezhnev', 'Brezhnev', 'Brezhnev', 'Brezhnev', 
                              'Brezhnev', 'Brezhnev', 'Gorbachev', 'Gorbachev'],
        'editor_in_chief': ['Editor1', 'Editor2', 'Editor1', 'Editor2', 
                            'Editor3', 'Editor4', 'Editor3', 'Editor4']
    }
    
    df = pd.DataFrame(example_data)
    df.to_csv(output_path, index=False)
    
    print(f"Created example CSV at: {output_path}")
    print("\nExpected CSV format:")
    print(df.to_string(index=False))
    print("\nRequired columns: journal, decade, general_secretary, english_author")
    print("Optional columns: title, year, editor_in_chief, translator")
    print("\nJournal values: 'IL' or 'Vsesvit'")
    print("Decade values: 1950s, 1960s, 1970s, 1980s, 1990s")
    print("General Secretary: Stalin, Khrushchev, Brezhnev, Andropov, Chernenko, Gorbachev")
    print("\nNote: Genre is no longer used in the model")
    
    return df


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("SOVIET LITERARY JOURNAL PUBLICATION PREDICTOR")
    print("Bayesian Network Approach")
    print("=" * 80)
    print()
    
    # THREE MODES OF OPERATION:
    # Mode 1: Prior knowledge (hand-coded probabilities)
    # Mode 2: Learn from data (MLE from CSV)
    # Mode 3: Create example CSV
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-example":
        # Mode 3: Create example CSV
        print("MODE 3: Creating example CSV file")
        print("=" * 80)
        print()
        output_path = sys.argv[2] if len(sys.argv) > 2 else "example_soviet_journals.csv"
        create_example_csv(output_path)
        
    elif len(sys.argv) > 1:
        # Mode 2: Learn from data
        data_path = sys.argv[1]
        print(f"MODE 2: Learning from data ({data_path})")
        print("=" * 80)
        print()
        
        try:
            bn = SovietJournalBayesianNetwork(learn_from_data=True, data_path=data_path)
            
            # Evaluate on test data
            results = bn.evaluate_on_data()
            
            # Analyze explaining away
            bn.analyze_explaining_away()
            
            # Show some predictions
            print("=" * 80)
            print("SAMPLE PREDICTIONS")
            print("=" * 80)
            print()
            
            # Get some test examples
            if bn.data is not None and len(bn.data) > 0:
                test_samples = bn.data.sample(n=min(5, len(bn.data)), random_state=42)
                
                print("Sample Predictions:")
                for idx, row in test_samples.iterrows():
                    author_name = row['english_author']
                    decade_name = bn.decades[int(row['decade_idx'])]
                    gs_name = bn.general_secretaries[int(row['gs_idx'])]
                    actual_journal = bn.journals[int(row['journal_idx'])]
                    
                    pub_score, pred_journal, conf = bn.predict_author_publishability(
                        author_name,
                        int(row['decade_idx']), 
                        int(row['gs_idx'])
                    )
                    
                    match = "✓" if pred_journal == actual_journal else "✗"
                    
                    print(f"\n{match} {author_name[:30]:30}")
                    print(f"   Era: {decade_name}, {gs_name}")
                    print(f"   Actual: {actual_journal:8} | Predicted: {pred_journal:8} (conf: {conf:.3f})")
                    print(f"   Publishability score: {pub_score:.3f}")
                
                # Show top authors for different eras
                print("\n" + "=" * 80)
                print("TOP AUTHORS BY ERA")
                print("=" * 80)
                
                print("\n1960s under Brezhnev:")
                top = bn.get_top_authors_for_era(decade_idx=1, gs_idx=2, top_k=5)
                for author, prob in top:
                    print(f"  {author:30} {prob:.4f}")
                
                print("\n1980s under Gorbachev:")
                top = bn.get_top_authors_for_era(decade_idx=3, gs_idx=5, top_k=5)
                for author, prob in top:
                    print(f"  {author:30} {prob:.4f}")
                
                # Predict for hypothetical authors
                print("\n" + "=" * 80)
                print("HYPOTHETICAL AUTHOR PREDICTIONS")
                print("=" * 80)
                
                # Test with known authors in different eras
                test_authors = bn.data['author_idx'].value_counts().head(3).index.tolist()
                for author_idx in test_authors:
                    author = bn.authors[author_idx]
                    print(f"\n{author}:")
                    for d_idx, decade in enumerate(bn.decades):
                        # Use most common GS for that decade
                        gs_idx = 2 if d_idx < 3 else 5  # Brezhnev or Gorbachev
                        pub_score, journal, conf = bn.predict_author_publishability(
                            author, d_idx, gs_idx
                        )
                        print(f"  {decade}: {journal:8} (pub: {pub_score:.3f}, conf: {conf:.3f})")
        
        except Exception as e:
            print(f"Error loading data: {e}")
            print("\nPlease ensure your CSV has the required columns:")
            print("  - english_author (American author name)")
            print("  - journal (values: 'IL' or 'Vsesvit')")
            print("  - decade (values: 1950s, 1960s, 1970s, 1980s, 1990s)")
            print("  - general_secretary (values: Stalin, Khrushchev, Brezhnev, Andropov, Chernenko, Gorbachev)")
            print("\nRun with --create-example to see the expected format:")
            print("  python bayes.py --create-example")
    
    else:
        # Mode 1: Prior knowledge
        print("MODE 1: Using prior knowledge (hand-coded probabilities)")
        print()
        print("Usage options:")
        print("  python bayes.py                    # Use prior knowledge (demo mode)")
        print("  python bayes.py <data.csv>         # Learn from your data")
        print("  python bayes.py --create-example   # Create example CSV file")
        print("=" * 80)
        print()
        
        # Initialize the Bayesian network with prior knowledge
        bn = SovietJournalBayesianNetwork(learn_from_data=False)
        
        print("\n" + "=" * 80)
        print("Note: Demo mode uses hand-coded probabilities.")
        print("To use with real data, run: python bayes.py your_data.csv")
        print("Your CSV must have an 'english_author' column.")
        print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)