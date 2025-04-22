import numpy as np
from agents.agent import Agent

class ThreatCreationAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "Threat Creation Agent"
        self._free_spaces = set((x, y) for x in range(self.size) for y in range(self.size))
        # For Player 1, the path is from top (y=0) to bottom (y=size-1)
        # For Player 2, the path is from left (x=0) to right (x=size-1)
        self.my_axis = 1 if self.player_number == 1 else 0
        self.my_start = 0
        self.my_end = self.size - 1
    
    def step(self):
        # Look for a move that creates threats
        threats = self._find_threats()
        
        if threats:
            # If there are threats from the opponent, block the most severe one
            opponent_threats = [t for t in threats if t['player'] == self.adv_number]
            if opponent_threats and (not any(t['player'] == self.player_number for t in threats) or 
                                     max(t['severity'] for t in opponent_threats) > 
                                     max(t['severity'] for t in threats if t['player'] == self.player_number)):
                move = max(opponent_threats, key=lambda x: x['severity'])['move']
            else:
                # Otherwise, play to strengthen our own threats
                own_threats = [t for t in threats if t['player'] == self.player_number]
                move = max(own_threats, key=lambda x: x['severity'])['move']
        else:
            # If no threats, create a new potential threat or take a strategic position
            move = self._create_new_threat()
        
        # Update internal state
        self._free_spaces.remove((move[0], move[1]))
        self.set_hex(self.player_number, move)
        return move
    
    def update(self, move_other_player):
        self.set_hex(self.adv_number, move_other_player)
        self._free_spaces.remove((move_other_player[0], move_other_player[1]))
    
    def _find_threats(self):
        """Find threats on the board - positions that would create significant advantage."""
        threats = []
        
        # Check each free position for potential threats
        for x, y in self._free_spaces:
            # Check if this position would create a threat for us
            my_threat_severity = self._evaluate_threat([x, y], self.player_number)
            if my_threat_severity > 0:
                threats.append({
                    'move': [x, y],
                    'player': self.player_number,
                    'severity': my_threat_severity
                })
            
            # Check if this position would create a threat for opponent
            opponent_threat_severity = self._evaluate_threat([x, y], self.adv_number)
            if opponent_threat_severity > 0:
                threats.append({
                    'move': [x, y],
                    'player': self.adv_number,
                    'severity': opponent_threat_severity
                })
        
        return threats
    
    def _evaluate_threat(self, move, player):
        """Evaluate the severity of a threat if the given player plays at the move."""
        # Make a temporary move to evaluate threat
        temp_grid = self._grid.copy()
        original_value = self._grid[move[1], move[0]]
        self._grid[move[1], move[0]] = player
        
        # Evaluate threat severity
        severity = 0
        
        # Check if this move creates a potential winning path
        can_win = self._has_potential_winning_path(player)
        if can_win:
            severity += 10
        
        # Check for bridge connection potential (pattern recognition)
        bridges = self._count_potential_bridges(move, player)
        severity += bridges * 2
        
        # Check for fork creation (multiple threat paths)
        forks = self._count_fork_threats(move, player)
        severity += forks * 5
        
        # Restore the grid
        self._grid[move[1], move[0]] = original_value
        
        return severity
    
    def _has_potential_winning_path(self, player):
        """Check if the player has a potential winning path."""
        # For simplicity, we'll just check if there's one continuous path
        # that's at least 1/3 of the way across the board
        axis = 1 if player == 1 else 0
        endpoint = self.size - 1
        
        # For player 1 (top-bottom), check paths from the top row
        # For player 2 (left-right), check paths from the leftmost column
        if player == 1:
            start_positions = [[x, 0] for x in range(self.size) if self.get_hex([x, 0]) == player]
        else:
            start_positions = [[0, y] for y in range(self.size) if self.get_hex([0, y]) == player]
        
        # Simple DFS to find the furthest progress along the main axis
        max_progress = 0
        for start in start_positions:
            visited = set()
            queue = [start]
            while queue:
                current = queue.pop(0)
                if current[axis] > max_progress:
                    max_progress = current[axis]
                visited.add(tuple(current))
                for neighbor in self.neighbors(current):
                    if (tuple(neighbor) not in visited and 
                        self.get_hex(neighbor) == player):
                        queue.append(neighbor)
        
        # Consider it a threat if we're at least 1/3 of the way across
        return max_progress >= endpoint / 3
    
    def _count_potential_bridges(self, move, player):
        """Count potential bridge formations from this move."""
        count = 0
        
        # A bridge typically has a gap of one hex between two friendly pieces
        # Check all neighbor pairs with one hex in between
        directions = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
        
        for i, dir1 in enumerate(directions):
            for j in range(i+1, len(directions)):
                dir2 = directions[j]
                # Check if both directions lead to the same player's pieces
                pos1 = [move[0] + dir1[0], move[1] + dir1[1]]
                pos2 = [move[0] + dir2[0], move[1] + dir2[1]]
                
                if (0 <= pos1[0] < self.size and 0 <= pos1[1] < self.size and
                    0 <= pos2[0] < self.size and 0 <= pos2[1] < self.size and
                    self.get_hex(pos1) == player and self.get_hex(pos2) == player):
                    count += 1
        
        return count
    
    def _count_fork_threats(self, move, player):
        """Count how many fork threats this move creates."""
        # A fork is when a move creates multiple threat paths
        # For simplicity, we'll count how many different empty neighbor pairs exist
        empty_neighbors = []
        for neighbor in self.neighbors(move):
            if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:
                if self.get_hex(neighbor) == 0:  # Empty hex
                    empty_neighbors.append(neighbor)
        
        # Count pairs of empty neighbors
        fork_count = 0
        for i in range(len(empty_neighbors)):
            for j in range(i+1, len(empty_neighbors)):
                # Check if these neighbors form a potential path
                # For simplicity, we'll just count all pairs
                fork_count += 1
        
        return fork_count
    
    def _create_new_threat(self):
        """Create a new threat if none exist."""
        best_score = float('-inf')
        best_move = None
        
        # Evaluate each free position
        for x, y in self._free_spaces:
            # Score based on position and connectivity
            score = self._position_score([x, y])
            
            if score > best_score:
                best_score = score
                best_move = [x, y]
        
        return best_move if best_move else list(self._free_spaces)[0]
    
    def _position_score(self, move):
        """Score a position based on strategic value."""
        score = 0
        
        # Positions closer to the center are generally better
        center = self.size // 2
        center_dist = abs(move[0] - center) + abs(move[1] - center)
        score -= center_dist  # Negative score for distance from center
        
        # If we're player 1 (top-bottom), prefer moves that advance vertically
        if self.player_number == 1:
            score += move[1]  # Higher score for positions further down
        # If we're player 2 (left-right), prefer moves that advance horizontally
        else:
            score += move[0]  # Higher score for positions further right
            
        # Bonus for moves that connect to our existing pieces
        for neighbor in self.neighbors(move):
            if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:
                if self.get_hex(neighbor) == self.player_number:
                    score += 2
        
        return score