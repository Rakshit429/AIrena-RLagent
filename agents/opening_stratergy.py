from agents.agent import Agent
class OpeningStrategyAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "OpeningStrategyAgent"
        self.moves_made = 0
        self.last_adv_move = None
        self.center = size // 2  # Center of the board
        
        # Opening book strategies (coordinates are [x, y])
        self.opening_moves = {
            # First move strategies
            'center': [self.center, self.center],
            'side_center': [0, self.center] if self.player_number == 1 else [self.center, 0],
            
            # Second move responses based on opponent's move
            'adjacent_to_opponent': None,  # Will be computed dynamically
            'diagonal_from_center': [self.center-1, self.center-1]
        }
        
        # For larger board sizes, we can pre-compute some strategic positions
        self.strategic_zones = self._compute_strategic_zones()

    def _compute_strategic_zones(self):
        """Calculate strategic zones based on board size and player number"""
        zones = {}
        # For player 1 (left-right), horizontal central line is valuable
        if self.player_number == 1:
            zones['central_path'] = [[x, self.center] for x in range(self.size)]
            # "Ladder" structure to climb vertically when needed
            zones['ladder_up'] = [[x, self.center - (x % 2)] for x in range(1, self.size-1, 2)]
            zones['ladder_down'] = [[x, self.center + (x % 2)] for x in range(1, self.size-1, 2)]
        # For player 2 (top-bottom), vertical central line is valuable
        else:
            zones['central_path'] = [[self.center, y] for y in range(self.size)]
            # "Ladder" structure to move horizontally when needed
            zones['ladder_left'] = [[self.center - (y % 2), y] for y in range(1, self.size-1, 2)]
            zones['ladder_right'] = [[self.center + (y % 2), y] for y in range(1, self.size-1, 2)]
        
        return zones

    def _find_adjacent_to_opponent(self):
        """Find a good adjacent position to opponent's last move"""
        if not self.last_adv_move:
            return None
        
        # Get all free adjacent positions
        adjacent_positions = []
        for n in self.neighbors(self.last_adv_move):
            if self.get_hex(n) == 0:  # Empty cell
                adjacent_positions.append(n)
        
        if not adjacent_positions:
            return None
            
        # For player 1 (horizontal), prefer positions to the right of opponent
        if self.player_number == 1:
            rightward = [pos for pos in adjacent_positions if pos[0] > self.last_adv_move[0]]
            if rightward:
                return rightward[0]
        # For player 2 (vertical), prefer positions below the opponent
        else:
            downward = [pos for pos in adjacent_positions if pos[1] > self.last_adv_move[1]]
            if downward:
                return downward[0]
                
        # If no preferred direction, pick any adjacent position
        return adjacent_positions[0] if adjacent_positions else None

    def _is_worth_blocking(self, position):
        """Determine if a position is worth blocking based on strategic value"""
        # For player 1, blocking positions that help opponent connect vertically
        if self.player_number == 1:
            # Check if this position is part of a vertical line of opponent pieces
            vertical_count = 1
            # Check above
            y = position[1] - 1
            while y >= 0 and self.get_hex([position[0], y]) == self.adv_number:
                vertical_count += 1
                y -= 1
            # Check below
            y = position[1] + 1
            while y < self.size and self.get_hex([position[0], y]) == self.adv_number:
                vertical_count += 1
                y += 1
            return vertical_count >= 2
        # For player 2, blocking positions that help opponent connect horizontally
        else:
            # Check if this position is part of a horizontal line of opponent pieces
            horizontal_count = 1
            # Check left
            x = position[0] - 1
            while x >= 0 and self.get_hex([x, position[1]]) == self.adv_number:
                horizontal_count += 1
                x -= 1
            # Check right
            x = position[0] + 1
            while x < self.size and self.get_hex([x, position[1]]) == self.adv_number:
                horizontal_count += 1
                x += 1
            return horizontal_count >= 2

    def step(self):
        """Decide the next move using opening strategies"""
        self.moves_made += 1
        
        # OPENING PHASE (first few moves)
        if self.moves_made <= 3:
            # First move: Choose a strategic opening
            if self.moves_made == 1:
                # On smaller boards, center start is strong
                if self.size <= 7:
                    if self.get_hex(self.opening_moves['center']) == 0:
                        return self.opening_moves['center']
                # On larger boards, side-center is often better for creating connections
                else:
                    if self.get_hex(self.opening_moves['side_center']) == 0:
                        return self.opening_moves['side_center']
            
            # Second/third moves: Respond to opponent or continue strategy
            elif self.last_adv_move:
                # Try to block if opponent has a strong position
                if self._is_worth_blocking(self.last_adv_move):
                    adjacent_pos = self._find_adjacent_to_opponent()
                    if adjacent_pos and self.get_hex(adjacent_pos) == 0:
                        return adjacent_pos
                
                # Otherwise continue with strategic development
                for zone_name, positions in self.strategic_zones.items():
                    for pos in positions:
                        if self.get_hex(pos) == 0:
                            return pos
        
        # MID GAME PHASE
        else:
            # Try to find a move that extends our current position
            own_positions = [[x, y] for x in range(self.size) for y in range(self.size) 
                            if self.get_hex([x, y]) == self.player_number]
            
            # For each of our positions, check neighbors
            for pos in own_positions:
                neighbors_list = self.neighbors(pos)
                # Prioritize neighbors that connect to other friendly pieces
                for n in neighbors_list:
                    if self.get_hex(n) == 0:  # Empty cell
                        # Check if this neighbor is also adjacent to another friendly piece
                        n_neighbors = self.neighbors(n)
                        has_another_friendly = False
                        for nn in n_neighbors:
                            if nn != pos and self.get_hex(nn) == self.player_number:
                                has_another_friendly = True
                                break
                        if has_another_friendly:
                            return n
                        
                # If no connecting neighbors, pick any neighbor
                for n in neighbors_list:
                    if self.get_hex(n) == 0:
                        return n
        
        # If no strategic moves found, pick any free position with some bias
        free_moves = self.free_moves()
        if not free_moves:
            return None
        
        # Bias toward moves in our direction of connection
        if self.player_number == 1:  # Horizontal player
            # Sort by x-coordinate (prioritize rightward movement)
            free_moves.sort(key=lambda pos: pos[0], reverse=True)
        else:  # Vertical player
            # Sort by y-coordinate (prioritize downward movement)
            free_moves.sort(key=lambda pos: pos[1], reverse=True)
            
        return free_moves[0]

    def update(self, move_other_player):
        """Update agent state with the opponent's move"""
        self.last_adv_move = move_other_player
        self.set_hex(self.adv_number, move_other_player)
        # Update the "adjacent to opponent" opening move
        self.opening_moves['adjacent_to_opponent'] = self._find_adjacent_to_opponent()