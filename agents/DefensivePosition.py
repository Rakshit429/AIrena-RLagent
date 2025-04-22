import numpy as np
from agents.agent import Agent

class DefensivePositioningAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "Defensive Positioning Agent"
        self._free_spaces = set((x, y) for x in range(self.size) for y in range(self.size))
        
        # Initialize defensive patterns based on player number
        if self.player_number == 1:  # top-bottom player
            self.defensive_axis = 0  # defend across the x-axis
            self.offensive_axis = 1  # attack along the y-axis
        else:  # left-right player
            self.defensive_axis = 1  # defend across the y-axis
            self.offensive_axis = 0  # attack along the x-axis
        
    def step(self):
        # First, check if there's an immediate defensive move needed
        defensive_move = self._find_defensive_move()
        if defensive_move:
            move = defensive_move
        else:
            # If no immediate defense needed, make a strategic offensive move
            move = self._make_offensive_move()
        
        # Update internal state
        self._free_spaces.remove((move[0], move[1]))
        self.set_hex(self.player_number, move)
        return move
    
    def update(self, move_other_player):
        self.set_hex(self.adv_number, move_other_player)
        self._free_spaces.remove((move_other_player[0], move_other_player[1]))
    
    def _find_defensive_move(self):
        """Find a move that blocks opponent's progress."""
        # Get all opponent's pieces
        opponent_pieces = []
        for x in range(self.size):
            for y in range(self.size):
                if self.get_hex([x, y]) == self.adv_number:
                    opponent_pieces.append([x, y])
        
        # If no opponent pieces, no need for defense
        if not opponent_pieces:
            return None
        
        # Calculate defensive scores for all free spaces
        defensive_scores = {}
        for x, y in self._free_spaces:
            score = self._calculate_defensive_score([x, y], opponent_pieces)
            defensive_scores[(x, y)] = score
        
        # Get the move with the highest defensive score
        if defensive_scores:
            best_defensive_move = max(defensive_scores.items(), key=lambda x: x[1])
            # Only make a defensive move if the score is high enough (urgent)
            if best_defensive_move[1] > 5:
                return [best_defensive_move[0][0], best_defensive_move[0][1]]
        
        return None
    
    def _calculate_defensive_score(self, move, opponent_pieces):
        """Calculate how good a move is defensively."""
        score = 0
        
        # Check how many opponent pieces are adjacent
        adjacent_opponents = 0
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.adv_number:
                adjacent_opponents += 1
                
        # Higher score for positions adjacent to multiple opponent pieces
        score += adjacent_opponents * 2
        
        # Higher score for positions that block opponent's progress along their winning axis
        for piece in opponent_pieces:
            # Calculate progress of this piece along opponent's axis
            opponent_progress = piece[self.defensive_axis]
            if self.adv_number == 1:  # opponent is top-bottom player
                opponent_progress = piece[1]  # measure y-axis progress
            else:  # opponent is left-right player
                opponent_progress = piece[0]  # measure x-axis progress
                
            # More points for blocking pieces that have made more progress
            progress_weight = opponent_progress / self.size
            if progress_weight > 0.5:  # Higher weight for pieces past halfway point
                score += 3 * progress_weight
        
        # Check if this move would block a potential connection
        score += self._evaluate_blocking_potential(move) * 3
        
        return score
    
    def _evaluate_blocking_potential(self, move):
        """Evaluate how well this move blocks potential connections."""
        blocking_value = 0
        
        # Make a temporary move to check impact
        original_value = self._grid[move[1], move[0]]
        self._grid[move[1], move[0]] = self.player_number
        
        # Check if this move blocks the formation of opponent bridges
        directions = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
        for i, dir1 in enumerate(directions):
            for j in range(i+1, len(directions)):
                dir2 = directions[j]
                pos1 = [move[0] + dir1[0], move[1] + dir1[1]]
                pos2 = [move[0] + dir2[0], move[1] + dir2[1]]
                
                if (0 <= pos1[0] < self.size and 0 <= pos1[1] < self.size and
                    0 <= pos2[0] < self.size and 0 <= pos2[1] < self.size):
                    if self.get_hex(pos1) == self.adv_number and self.get_hex(pos2) == self.adv_number:
                        # This move blocks a potential bridge
                        blocking_value += 4
        
        # Restore the grid
        self._grid[move[1], move[0]] = original_value
        
        return blocking_value
    
    def _make_offensive_move(self):
        """Make a strategic offensive move."""
        best_score = float('-inf')
        best_move = None
        
        for x, y in self._free_spaces:
            # Calculate offensive and defensive components
            offensive_score = self._calculate_offensive_score([x, y])
            defensive_component = self._calculate_defensive_component([x, y])
            
            # Combined score with bias toward defense
            score = offensive_score + defensive_component * 1.5
            
            if score > best_score:
                best_score = score
                best_move = [x, y]
        
        return best_move if best_move else list(self._free_spaces)[0]
    
    def _calculate_offensive_score(self, move):
        """Calculate how good a move is offensively."""
        score = 0
        
        # Positions closer to our winning edge are better
        if self.player_number == 1:  # top-bottom player
            progress = move[1] / self.size  # y-axis progress
            edge_x = move[0] / self.size  # horizontal position
            # Prefer central x-positions
            score += 1 - 2 * abs(edge_x - 0.5)
        else:  # left-right player
            progress = move[0] / self.size  # x-axis progress
            edge_y = move[1] / self.size  # vertical position
            # Prefer central y-positions
            score += 1 - 2 * abs(edge_y - 0.5)
            
        # Score based on progress toward goal
        score += progress * 2
        
        # Bonus for moves that connect to our existing pieces
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.player_number:
                score += 1.5
                
        return score
    
    def _calculate_defensive_component(self, move):
        """Calculate defensive value as a component of an offensive move."""
        score = 0
        
        # Check how well this move blocks opponent's central path
        if self.player_number == 1:  # We are top-bottom player
            # Blocking opponent's central horizontal path
            center_y = self.size // 2
            vertical_dist = abs(move[1] - center_y)
            # Moves closer to central horizontal line are better defensively
            score += 1 - vertical_dist / self.size
        else:  # We are left-right player
            # Blocking opponent's central vertical path
            center_x = self.size // 2
            horizontal_dist = abs(move[0] - center_x)
            # Moves closer to central vertical line are better defensively
            score += 1 - horizontal_dist / self.size
            
        # Defensive value of blocking opponent pieces
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.adv_number:
                score += 0.75
                
        return score