from agents.agent import Agent
class TemplatePatternAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "TemplatePatternAgent"
        self.last_adv_move = None
        self.move_history = []
        self.opponent_moves = []
        self.current_phase = "opening"
        
        # Initialize phase transition thresholds
        self.mid_game_threshold = size * 2
        self.end_game_threshold = size * size - (size * 2)

    def step(self):
        """Template method that defines the algorithm structure"""
        # Update game phase based on move count
        self._update_game_phase()
        
        # Step 1: Analyze the current board state
        board_analysis = self._analyze_board()
        
        # Step 2: Generate candidate moves
        candidate_moves = self._generate_candidate_moves(board_analysis)
        
        # Step 3: Evaluate candidate moves
        evaluated_moves = self._evaluate_moves(candidate_moves, board_analysis)
        
        # Step 4: Select the best move
        best_move = self._select_best_move(evaluated_moves)
        
        # Step 5: Record the move for future analysis
        if best_move:
            self.move_history.append(best_move)
            
        return best_move

    def update(self, move_other_player):
        """Update agent state with the opponent's move"""
        self.last_adv_move = move_other_player
        self.opponent_moves.append(move_other_player)
        self.set_hex(self.adv_number, move_other_player)

    def _update_game_phase(self):
        """Update the current game phase based on the number of moves made"""
        total_moves = len(self.move_history) + len(self.opponent_moves)
        
        if total_moves < self.mid_game_threshold:
            self.current_phase = "opening"
        elif total_moves < self.end_game_threshold:
            self.current_phase = "mid_game"
        else:
            self.current_phase = "end_game"

    def _analyze_board(self):
        """Analyze the current board state and return a dictionary of analysis results"""
        analysis = {
            "phase": self.current_phase,
            "own_positions": [],
            "adv_positions": [],
            "own_territory": 0,
            "adv_territory": 0,
            "critical_points": [],
            "connection_points": [],
            "blocking_points": []
        }
        
        # Collect positions for both players
        for x in range(self.size):
            for y in range(self.size):
                pos = [x, y]
                if self.get_hex(pos) == self.player_number:
                    analysis["own_positions"].append(pos)
                elif self.get_hex(pos) == self.adv_number:
                    analysis["adv_positions"].append(pos)
        
        # Identify critical points (junction points that connect multiple pieces)
        for pos in self.free_moves():
            own_neighbors = 0
            adv_neighbors = 0
            
            for n in self.neighbors(pos):
                if self.get_hex(n) == self.player_number:
                    own_neighbors += 1
                elif self.get_hex(n) == self.adv_number:
                    adv_neighbors += 1
            
            # Add to critical points if this position connects to multiple pieces
            if own_neighbors >= 2:
                analysis["connection_points"].append({"position": pos, "strength": own_neighbors})
            
            # Add to blocking points if this position blocks opponent connections
            if adv_neighbors >= 2:
                analysis["blocking_points"].append({"position": pos, "strength": adv_neighbors})
            
            # Add to critical points if this is valuable for either player
            if own_neighbors + adv_neighbors >= 2:
                analysis["critical_points"].append({
                    "position": pos, 
                    "own_neighbors": own_neighbors,
                    "adv_neighbors": adv_neighbors,
                    "value": own_neighbors * 1.5 + adv_neighbors
                })
        
        # Calculate territory control metrics
        analysis["own_territory"] = self._calculate_territory(self.player_number)
        analysis["adv_territory"] = self._calculate_territory(self.adv_number)
        
        return analysis

    def _calculate_territory(self, player):
        """Calculate approximate territory control for a player"""
        territory = 0
        
        for x in range(self.size):
            for y in range(self.size):
                pos = [x, y]
                if self.get_hex(pos) == player:
                    # Count direct control
                    territory += 1
                    # Count influence on adjacent empty cells
                    for n in self.neighbors(pos):
                        if self.get_hex(n) == 0:
                            territory += 0.3
        
        return territory

    def _generate_candidate_moves(self, analysis):
        """Generate candidate moves based on the current phase and board analysis"""
        candidates = []
        
        # Phase-specific move generation
        if analysis["phase"] == "opening":
            candidates = self._generate_opening_moves(analysis)
        elif analysis["phase"] == "mid_game":
            candidates = self._generate_mid_game_moves(analysis)
        else:  # end_game
            candidates = self._generate_end_game_moves(analysis)
        
        # Add critical points to candidates if not already included
        for point in analysis["critical_points"]:
            if point["position"] not in candidates:
                candidates.append(point["position"])
        
        # Add some random valid moves for diversity
        free_moves = self.free_moves()
        if free_moves:
            import random
            random_samples = min(3, len(free_moves))
            for _ in range(random_samples):
                random_move = random.choice(free_moves)
                if random_move not in candidates:
                    candidates.append(random_move)
        
        return candidates

    def _generate_opening_moves(self, analysis):
        """Generate opening phase moves"""
        candidates = []
        center = self.size // 2
        
        # Strong opening moves
        opening_positions = [
            [center, center],  # Center
            [center-1, center-1], [center+1, center+1],  # Diagonals
            [center-1, center+1], [center+1, center-1],  # Diagonals
        ]
        
        # For player 1 (horizontal connection), add left-center and right-center
        if self.player_number == 1:
            opening_positions.extend([
                [0, center],  # Left edge center
                [self.size-1, center]  # Right edge center
            ])
        # For player 2 (vertical connection), add top-center and bottom-center
        else:
            opening_positions.extend([
                [center, 0],  # Top edge center
                [center, self.size-1]  # Bottom edge center
            ])
        
        for pos in opening_positions:
            if 0 <= pos[0] < self.size and 0 <= pos[1] < self.size and self.get_hex(pos) == 0:
                candidates.append(pos)
        
        return candidates

    def _generate_mid_game_moves(self, analysis):
        """Generate mid-game phase moves"""
        candidates = []
        
        # Prioritize connection points
        for point in analysis["connection_points"]:
            candidates.append(point["position"])
        
        # Add blocking points if opponent is threatening
        for point in analysis["blocking_points"]:
            if point["strength"] >= 2:  # Only block significant threats
                candidates.append(point["position"])
        
        # Add positions that extend toward goal
        if self.player_number == 1:  # Horizontal connection
            for pos in analysis["own_positions"]:
                # Look for neighbors to the right
                for n in self.neighbors(pos):
                    if n[0] > pos[0] and self.get_hex(n) == 0 and n not in candidates:
                        candidates.append(n)
        else:  # Vertical connection
            for pos in analysis["own_positions"]:
                # Look for neighbors below
                for n in self.neighbors(pos):
                    if n[1] > pos[1] and self.get_hex(n) == 0 and n not in candidates:
                        candidates.append(n)
        
        return candidates

    def _generate_end_game_moves(self, analysis):
        """Generate end-game phase moves"""
        candidates = []
        
        # In end game, prioritize completing connections
        # For player 1, focus on rightward connections
        if self.player_number == 1:
            # Find pieces closest to right edge
            rightmost_pieces = sorted(analysis["own_positions"], key=lambda pos: pos[0], reverse=True)
            if rightmost_pieces:
                # Add empty neighbors of rightmost pieces
                for pos in rightmost_pieces[:3]:  # Consider top 3 rightmost pieces
                    for n in self.neighbors(pos):
                        if self.get_hex(n) == 0 and n not in candidates:
                            candidates.append(n)
        # For player 2, focus on downward connections
        else:
            # Find pieces closest to bottom edge
            bottommost_pieces = sorted(analysis["own_positions"], key=lambda pos: pos[1], reverse=True)
            if bottommost_pieces:
                # Add empty neighbors of bottommost pieces
                for pos in bottommost_pieces[:3]:  # Consider top 3 bottommost pieces
                    for n in self.neighbors(pos):
                        if self.get_hex(n) == 0 and n not in candidates:
                            candidates.append(n)
        
        # Add critical blocking positions
        for point in analysis["blocking_points"]:
            if point["strength"] >= 2:  # Only block significant threats
                candidates.append(point["position"])
        
        return candidates

    def _evaluate_moves(self, candidate_moves, analysis):
        """Evaluate each candidate move and return a list of (move, score) tuples"""
        evaluated_moves = []
        
        for move in candidate_moves:
            # Skip if the move is invalid
            if not (0 <= move[0] < self.size and 0 <= move[1] < self.size) or self.get_hex(move) != 0:
                continue
                
            # Calculate a score for this move
            score = self._evaluate_single_move(move, analysis)
            evaluated_moves.append((move, score))
        
        return evaluated_moves

    def _evaluate_single_move(self, move, analysis):
        """Evaluate a single move and return a score"""
        score = 0
        
        # Evaluate based on the current phase
        if analysis["phase"] == "opening":
            score += self._evaluate_opening_move(move, analysis)
        elif analysis["phase"] == "mid_game":
            score += self._evaluate_mid_game_move(move, analysis)
        else:  # end_game
            score += self._evaluate_end_game_move(move, analysis)
        
        # Common evaluation criteria
        
        # Connection strength - how many of our pieces this connects to
        own_neighbors = 0
        adv_neighbors = 0
        for n in self.neighbors(move):
            if self.get_hex(n) == self.player_number:
                own_neighbors += 1
            elif self.get_hex(n) == self.adv_number:
                adv_neighbors += 1
        
        score += own_neighbors * 10  # Strong bonus for connecting to own pieces
        
        # Position relative to goal direction
        if self.player_number == 1:  # Horizontal connection
            # Reward positions closer to right edge
            score += move[0] * 2
            # Extra reward for being in the middle row
            middle_row = self.size // 2
            score += (self.size - abs(move[1] - middle_row)) * 0.5
        else:  # Vertical connection
            # Reward positions closer to bottom edge
            score += move[1] * 2
            # Extra reward for being in the middle column
            middle_col = self.size // 2
            score += (self.size - abs(move[0] - middle_col)) * 0.5
        
        # Blocking evaluation
        # More points for blocking opponent connections
        if adv_neighbors >= 2:
            score += adv_neighbors * 5
            
        # Check if this move blocks a key opponent path
        if self._blocks_opponent_path(move):
            score += 15
            
        # Check if this move creates a virtual connection
        if self._creates_virtual_connection(move):
            score += 12
            
        return score

    def _evaluate_opening_move(self, move, analysis):
        """Evaluate an opening move"""
        score = 0
        center = self.size // 2
        
        # Favor center and near-center positions in opening
        distance_to_center = abs(move[0] - center) + abs(move[1] - center)
        score += (self.size - distance_to_center) * 2
        
        # For player 1, slightly favor positions on the left side
        if self.player_number == 1 and move[0] < center:
            score += 3
        # For player 2, slightly favor positions on the top side
        elif self.player_number == 2 and move[1] < center:
            score += 3
            
        return score

    def _evaluate_mid_game_move(self, move, analysis):
        """Evaluate a mid-game move"""
        score = 0
        
        # Check if this move is in our critical points list
        for critical in analysis["critical_points"]:
            if critical["position"] == move:
                score += critical["value"] * 5
                break
        
        # Check if this move extends our territory
        territory_value = 0
        for n in self.neighbors(move):
            if self.get_hex(n) == self.player_number:
                territory_value += 3
        score += territory_value
        
        return score

    def _evaluate_end_game_move(self, move, analysis):
        """Evaluate an end-game move"""
        score = 0
        
        # In end game, highly value moves that complete connections
        # Simulate making this move and check if it creates a win
        self._grid[move[1], move[0]] = self.player_number
        if self.check_win(self.player_number):
            score += 1000  # Huge bonus for winning moves
        
        # Check if this move creates a strong connection to goal
        if self._creates_connection_to_goal(move):
            score += 50
            
        # Undo the simulation
        self._grid[move[1], move[0]] = 0
        
        return score

    def _blocks_opponent_path(self, move):
        """Check if this move blocks an important opponent path"""
        # Temporarily place our piece
        self._grid[move[1], move[0]] = self.player_number
        
        # For player 1, check if we're blocking a vertical connection
        # For player 2, check if we're blocking a horizontal connection
        is_blocking = False
        
        if self.player_number == 1:  # We're trying to block vertical connections
            # Check if there are opponent pieces both above and below
            has_above = False
            has_below = False
            
            # Check above
            y = move[1] - 1
            while y >= 0:
                if self.get_hex([move[0], y]) == self.adv_number:
                    has_above = True
                    break
                y -= 1
                
            # Check below
            y = move[1] + 1
            while y < self.size:
                if self.get_hex([move[0], y]) == self.adv_number:
                    has_below = True
                    break
                y += 1
                
            is_blocking = has_above and has_below
                
        else:  # We're trying to block horizontal connections
            # Check if there are opponent pieces both left and right
            has_left = False
            has_right = False
            
            # Check left
            x = move[0] - 1
            while x >= 0:
                if self.get_hex([x, move[1]]) == self.adv_number:
                    has_left = True
                    break
                x -= 1
                
            # Check right
            x = move[0] + 1
            while x < self.size:
                if self.get_hex([x, move[1]]) == self.adv_number:
                    has_right = True
                    break
                x += 1
                
            is_blocking = has_left and has_right
        
        # Undo our temporary placement
        self._grid[move[1], move[0]] = 0
        
        return is_blocking

    def _creates_virtual_connection(self, move):
        """Check if this move creates a virtual connection between our pieces"""
        # Temporarily place our piece
        self._grid[move[1], move[0]] = self.player_number
        
        # Get all pairs of our pieces
        our_pieces = []
        for x in range(self.size):
            for y in range(self.size):
                if self.get_hex([x, y]) == self.player_number:
                    our_pieces.append([x, y])
        
        # Check if there's any pair that has exactly 2 common empty neighbors
        creates_connection = False
        for i, pos1 in enumerate(our_pieces):
            for pos2 in our_pieces[i+1:]:
                # Skip if they're neighbors already
                if any(n[0] == pos2[0] and n[1] == pos2[1] for n in self.neighbors(pos1)):
                    continue
                    
                # Count common empty neighbors
                common_empty_neighbors = 0
                for n1 in self.neighbors(pos1):
                    if self.get_hex(n1) == 0:
                        for n2 in self.neighbors(pos2):
                            if n1[0] == n2[0] and n1[1] == n2[1] and self.get_hex(n2) == 0:
                                common_empty_neighbors += 1
                
                if common_empty_neighbors == 2:  # Bridge virtual connection
                    creates_connection = True
                    break
            
            if creates_connection:
                break
        
        # Undo our temporary placement
        self._grid[move[1], move[0]] = 0
        
        return creates_connection

    def _creates_connection_to_goal(self, move):
        """Check if this move creates a strong connection to our goal"""
        # For player 1, check connection to right edge
        # For player 2, check connection to bottom edge
        
        if self.player_number == 1:
            # If move is already at right edge
            if move[0] == self.size - 1:
                return True
                
            # Check if any neighbor is at right edge
            for n in self.neighbors(move):
                if n[0] == self.size - 1 and self.get_hex(n) == self.player_number:
                    return True
        else:
            # If move is already at bottom edge
            if move[1] == self.size - 1:
                return True
                
            # Check if any neighbor is at bottom edge
            for n in self.neighbors(move):
                if n[1] == self.size - 1 and self.get_hex(n) == self.player_number:
                    return True
        
        return False

    def _select_best_move(self, evaluated_moves):
        """Select the best move from the evaluated moves"""
        if not evaluated_moves:
            free_moves = self.free_moves()
            if free_moves:
                return free_moves[0]
            return None
        
        # Sort moves by score (descending)
        evaluated_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Return the move with the highest score
        return evaluated_moves[0][0]