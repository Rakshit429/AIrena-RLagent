from agents.agent import Agent
import copy
class VirtualConnectionAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "VirtualConnectionAgent"
        self.last_adv_move = None
        
        # Virtual connection types
        self.connection_types = {
            'bridge': self._is_bridge,
            'zigzag': self._is_zigzag,
            'ladder': self._is_ladder
        }
        
        # Connection values (higher is better)
        self.connection_values = {
            'bridge': 3,
            'zigzag': 2,
            'ladder': 4
        }

    def _is_bridge(self, pos1, pos2):
        """Check if two positions can form a bridge virtual connection"""
        # A bridge is formed when two cells share two common neighbors
        common_neighbors = []
        for n1 in self.neighbors(pos1):
            if self.get_hex(n1) == 0:  # Empty cell
                for n2 in self.neighbors(pos2):
                    if n1[0] == n2[0] and n1[1] == n2[1] and self.get_hex(n2) == 0:
                        common_neighbors.append(n1)
        
        return len(common_neighbors) >= 2, common_neighbors

    def _is_zigzag(self, pos1, pos2):
        """Check if two positions can form a zigzag virtual connection"""
        # A zigzag is formed when two cells have a common neighbor with another common neighbor
        for n1 in self.neighbors(pos1):
            if self.get_hex(n1) == 0:  # Empty cell
                for n2 in self.neighbors(pos2):
                    if self.get_hex(n2) == 0 and n1 != n2:  # Different empty cells
                        # Check if n1 and n2 are neighbors
                        if any(n[0] == n2[0] and n[1] == n2[1] for n in self.neighbors(n1)):
                            return True, [n1, n2]
        return False, []

    def _is_ladder(self, pos1, pos2):
        """Check if two positions can form a ladder virtual connection"""
        # A ladder is formed when there are multiple alternative paths between two cells
        paths = []
        
        # Check all empty neighbors of pos1
        for n1 in self.neighbors(pos1):
            if self.get_hex(n1) == 0:  # Empty neighbor of pos1
                # Check if this neighbor has another empty neighbor that's also 
                # a neighbor of pos2 (forming a 3-cell path)
                for n1_neighbor in self.neighbors(n1):
                    if (self.get_hex(n1_neighbor) == 0 and n1_neighbor != pos1 and 
                        any(n[0] == n1_neighbor[0] and n[1] == n1_neighbor[1] for n in self.neighbors(pos2))):
                        paths.append([n1, n1_neighbor])
                        
                # Also check if this neighbor directly connects to pos2
                if any(n[0] == pos2[0] and n[1] == pos2[1] for n in self.neighbors(n1)):
                    paths.append([n1])
        
        return len(paths) >= 2, paths

    def _find_virtual_connections(self):
        """Find all potential virtual connections between owned pieces"""
        connections = []
        owned_positions = [[x, y] for x in range(self.size) for y in range(self.size)
                          if self.get_hex([x, y]) == self.player_number]
        
        # Compare each pair of owned positions
        for i, pos1 in enumerate(owned_positions):
            for pos2 in owned_positions[i+1:]:
                # Skip adjacent positions - they're already connected
                if any(n[0] == pos2[0] and n[1] == pos2[1] for n in self.neighbors(pos1)):
                    continue
                    
                # Check for each connection type
                for conn_type, check_function in self.connection_types.items():
                    is_connection, connecting_cells = check_function(pos1, pos2)
                    if is_connection:
                        connections.append({
                            'type': conn_type,
                            'positions': [pos1, pos2],
                            'connecting_cells': connecting_cells,
                            'value': self.connection_values[conn_type]
                        })
        
        return connections
    def _find_critical_cell(self, connections):
        """Find the most critical cell to place - one that's part of multiple connections"""
        if not connections:
            return None
            
        # Count how many times each cell appears in connections
        cell_counts = {}
        
        for conn in connections:
            # The structure of connecting_cells varies by connection type
            connecting_cells = conn['connecting_cells']
            
            # Handle different structures based on connection type
            if conn['type'] == 'bridge':
                # Bridge: connecting_cells is a list of individual cells
                cells_to_process = connecting_cells
            elif conn['type'] == 'zigzag':
                # Zigzag: connecting_cells is a list [n1, n2]
                cells_to_process = connecting_cells
            elif conn['type'] == 'ladder':
                # Ladder: connecting_cells is a list of paths, flatten it
                cells_to_process = []
                for path in connecting_cells:
                    cells_to_process.extend(path)
            else:
                # Unknown type, skip
                continue
                
            # Now process each cell
            for cell in cells_to_process:
                # Convert cell to tuple for use as dict key
                cell_tuple = tuple(cell)
                
                if cell_tuple not in cell_counts:
                    cell_counts[cell_tuple] = {'count': 0, 'value': 0}
                cell_counts[cell_tuple]['count'] += 1
                cell_counts[cell_tuple]['value'] += conn['value']
        
        # Find cell with highest count or value
        best_cell = None
        best_score = -1
        for cell_tuple, data in cell_counts.items():
            # Score combines count and value
            score = data['count'] * 2 + data['value']
            if score > best_score:
                best_score = score
                best_cell = list(cell_tuple)
                
        return best_cell
    
    def _evaluate_position_connectivity(self, pos):
        """Evaluate how well a position connects existing pieces"""
        score = 0
        
        # Check if this position helps connect toward our goal
        if self.player_number == 1:  # Horizontal connection (left to right)
            # Higher score for positions further to the right
            score += pos[0] / self.size
        else:  # Vertical connection (top to bottom)
            # Higher score for positions further to the bottom
            score += pos[1] / self.size
            
        # Check how many of our pieces this connects to
        own_neighbors = 0
        for n in self.neighbors(pos):
            if self.get_hex(n) == self.player_number:
                own_neighbors += 1
                
        score += own_neighbors * 0.5
        
        # Bonus for connecting to pieces that are already near goals
        for n in self.neighbors(pos):
            if self.get_hex(n) == self.player_number:
                if self.player_number == 1 and n[0] == self.size - 1:  # Right edge
                    score += 2
                elif self.player_number == 2 and n[1] == self.size - 1:  # Bottom edge
                    score += 2
                    
        return score

    def _can_block_opponent_connection(self, pos):
        """Check if placing at this position blocks an opponent's strong connection"""
        # Temporarily place our piece and check if it breaks opponent connections
        self._grid[pos[1], pos[0]] = self.player_number
        
        # Create a temporary agent to represent opponent's view
        opponent_agent = Agent(self.size, self.adv_number, self.player_number)  
        opponent_agent._grid = copy.deepcopy(self._grid)
        
        # Gather all opponent positions
        opponent_positions = [[x, y] for x in range(self.size) for y in range(self.size)
                            if opponent_agent.get_hex([x, y]) == self.adv_number]
        
        # Check if we're blocking a path
        blocking_value = 0
        
        # For player 1 (horizontal), check if we're blocking a vertical path
        if self.player_number == 1:
            # Check if there are opponent pieces both above and below this position
            has_above = False
            has_below = False
            for opp_pos in opponent_positions:
                if opp_pos[0] == pos[0]:  # Same column
                    if opp_pos[1] < pos[1]:  # Above
                        has_above = True
                    elif opp_pos[1] > pos[1]:  # Below
                        has_below = True
            if has_above and has_below:
                blocking_value = 5  # High value for blocking vertical connection
                
        # For player 2 (vertical), check if we're blocking a horizontal path
        else:
            # Check if there are opponent pieces both left and right of this position
            has_left = False
            has_right = False
            for opp_pos in opponent_positions:
                if opp_pos[1] == pos[1]:  # Same row
                    if opp_pos[0] < pos[0]:  # Left
                        has_left = True
                    elif opp_pos[0] > pos[0]:  # Right
                        has_right = True
            if has_left and has_right:
                blocking_value = 5  # High value for blocking horizontal connection
        
        # Undo our temporary placement
        self._grid[pos[1], pos[0]] = 0
        
        return blocking_value

    def step(self):
        """Decide the next move based on virtual connections"""
        # Find potential virtual connections
        connections = self._find_virtual_connections()
        
        # If we have potential connections, try to secure them
        critical_cell = self._find_critical_cell(connections)
        if critical_cell and self.get_hex(critical_cell) == 0:
            return critical_cell
        
        # If no critical cell from connections, evaluate all possible moves
        best_move = None
        best_score = -float('inf')
        
        for move in self.free_moves():
            # Score based on connectivity
            conn_score = self._evaluate_position_connectivity(move)
            
            # Score based on blocking opponent
            block_score = self._can_block_opponent_connection(move)
            
            # Combine scores
            total_score = conn_score + block_score
            
            if total_score > best_score:
                best_score = total_score
                best_move = move
        
        if best_move:
            return best_move
            
        # Fall back to a simple strategy if nothing else works
        free_moves = self.free_moves()
        if not free_moves:
            return None
            
        # For simplicity, pick center position if available
        center = self.size // 2
        if self.get_hex([center, center]) == 0:
            return [center, center]
            
        return free_moves[0]

    def update(self, move_other_player):
        """Update agent state with the opponent's move"""
        self.last_adv_move = move_other_player
        self.set_hex(self.adv_number, move_other_player)