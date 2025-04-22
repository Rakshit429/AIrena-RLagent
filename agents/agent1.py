import numpy as np
from agents.agent import Agent
import random
from collections import defaultdict, deque
import os
import math

class CustomPlayer(Agent):
    def __init__(self, size, player_number, adv_number, alpha=0.2, gamma=0.95, epsilon=0, 
                 epsilon_decay=0.9999, epsilon_min=0, load_q_table_path="training/saved_models/hex_rl_agent_5x5_all_opponents_final"):
        super().__init__(size, player_number, adv_number)
        self.name = "Enhanced RL Agent with Dual Q-Tables"
        # Learning parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Slower decay
        self.epsilon_min = epsilon_min  # Minimum exploration rate

        # Separate Q-tables for player 1 and player 2
        self.q_table_p1 = defaultdict(float)  # Q-table when agent is player 1
        self.q_table_p2 = defaultdict(float)  # Q-table when agent is player 2

        # Experience replay buffers (separate for each player role)
        self.replay_buffer_p1 = deque(maxlen=10000)
        self.replay_buffer_p2 = deque(maxlen=10000)
        self.replay_batch_size = 32

        # Opening book for common starting moves (for Hex, corners are strong)
        self.opening_book = [[0, 0], [0, size-1], [size-1, 0], [size-1, size-1]]
        self.opening_book_moves_left = 1  # Use opening book for first move only

        # Load existing Q-tables if provided
        if load_q_table_path and os.path.exists(load_q_table_path + "_p1.npy"):
            self.load_q_table(load_q_table_path)
            print(f"Loaded Q-tables from {load_q_table_path}")

        # Track the game state history for updating Q-values
        self.state_history = []
        
        # Track game moves for intermediate rewards
        self.moves_counter = 0
        
        # Track the last state for opponent move learning
        self.last_state = None
        self.last_move = None

    def get_state_key(self):
        """Convert the grid to a string representation for use as a dictionary key"""
        return str(self._grid.flatten().tolist())

    def get_action_key(self, action):
        """Convert an action to a string representation"""
        return f"{action[0]},{action[1]}"

    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair based on current player number"""
        state_key = state
        action_key = self.get_action_key(action)
        
        # Use the appropriate Q-table based on player number
        if self.player_number == 1:
            return self.q_table_p1.get((state_key, action_key), 0.0)
        else:  # player_number == 2
            return self.q_table_p2.get((state_key, action_key), 0.0)

    def update_q_value(self, state, action, new_value):
        """Update Q-value for a state-action pair in the appropriate Q-table"""
        state_key = state
        action_key = self.get_action_key(action)
        
        # Update the appropriate Q-table based on player number
        if self.player_number == 1:
            self.q_table_p1[(state_key, action_key)] = new_value
        else:  # player_number == 2
            self.q_table_p2[(state_key, action_key)] = new_value

    def choose_action(self):
        """Choose an action using epsilon-greedy policy with UCB exploration"""
        available_moves = self.free_moves()
        
        # If opening book moves are available, use them
        if self.moves_counter < self.opening_book_moves_left:
            for move in self.opening_book:
                if move in available_moves:
                    return move
        
        # If no moves are available, return None
        if not available_moves:
            return None

        current_state = self.get_state_key()
        
        # With probability epsilon, choose a random action (exploration)
        if random.random() < self.epsilon:
            return random.choice(available_moves)

        # Otherwise, choose the best action based on Q-values (exploitation)
        best_value = -float('inf')
        best_actions = []
        
        # UCB exploration factor (c is exploration parameter)
        c = 1.0
        
        # Choose appropriate visit counts dictionary based on player number
        if self.player_number == 1:
            visit_counts = defaultdict(int)
        else:
            visit_counts = defaultdict(int)
            
        total_visits = sum(visit_counts.values()) + 1  # Avoid division by zero
        
        for action in available_moves:
            # Get base Q-value
            q_value = self.get_q_value(current_state, action)
            
            # Add UCB exploration bonus
            action_key = self.get_action_key(action)
            visit_count = visit_counts.get((current_state, action_key), 0) + 1
            exploration_bonus = c * math.sqrt(math.log(total_visits) / visit_count)
            
            # Combined value with exploration bonus
            value = q_value + exploration_bonus

            if value > best_value:
                best_value = value
                best_actions = [action]
            elif value == best_value:
                best_actions.append(action)

        # If multiple actions have the same value, choose randomly among them
        return random.choice(best_actions)

    def calculate_connectivity_score(self):
        """Calculate connectivity score for both players"""
        player_score = self._connectivity_value(self.player_number)
        opponent_score = self._connectivity_value(self.adv_number)
        return player_score - opponent_score
    
    def _connectivity_value(self, player):
        """Calculate connectivity value for a player based on connected components"""
        if player == 1:  # Horizontal player
            start_edge = [[0, y] for y in range(self.size)]
            end_edge = [[self.size-1, y] for y in range(self.size)]
        else:  # Vertical player
            start_edge = [[x, 0] for x in range(self.size)]
            end_edge = [[x, self.size-1] for x in range(self.size)]
        
        # Find player's stones
        player_stones = []
        for x in range(self.size):
            for y in range(self.size):
                if self.get_hex([x, y]) == player:
                    player_stones.append([x, y])
        
        # If no stones, return 0
        if not player_stones:
            return 0
        
        # Find connected components
        components = []
        visited = set()
        
        for stone in player_stones:
            if tuple(stone) in visited:
                continue
                
            # Start new component
            component = []
            queue = [stone]
            while queue:
                current = queue.pop(0)
                if tuple(current) in visited:
                    continue
                    
                visited.add(tuple(current))
                component.append(current)
                
                # Add neighbors of same player
                for neighbor in self.neighbors(current):
                    if self.get_hex(neighbor) == player and tuple(neighbor) not in visited:
                        queue.append(neighbor)
            
            components.append(component)
        
        # Evaluate components
        score = 0
        for component in components:
            # Check if component connects to edges
            connects_start = any(tuple(stone) in [tuple(edge) for edge in start_edge] for stone in component)
            connects_end = any(tuple(stone) in [tuple(edge) for edge in end_edge] for stone in component)
            
            # Score based on size and edge connections
            component_score = len(component)
            if connects_start:
                component_score *= 1.5
            if connects_end:
                component_score *= 1.5
            if connects_start and connects_end:  # This means win
                component_score *= 10
                
            score += component_score
            
        return score

    def step(self):
        """Choose and execute an action"""
        # Save the current state
        current_state = self.get_state_key()
        
        # Calculate current connectivity score for intermediate rewards
        pre_move_score = self.calculate_connectivity_score()

        # Choose an action
        move = self.choose_action()

        # Safety check
        if move is None:
            available_moves = self.free_moves()
            if available_moves:
                move = available_moves[0]
            else:
                print("Warning: No valid moves available")
                return [0, 0]  # Fallback

        # Execute the action
        self.set_hex(self.player_number, move)
        self.moves_counter += 1
        
        # Calculate new connectivity score for intermediate rewards
        post_move_score = self.calculate_connectivity_score()
        immediate_reward = post_move_score - pre_move_score
        
        # Add to appropriate experience replay buffer
        next_state = self.get_state_key()
        if self.player_number == 1:
            self.replay_buffer_p1.append((current_state, move, immediate_reward, next_state))
        else:
            self.replay_buffer_p2.append((current_state, move, immediate_reward, next_state))

        # Save state-action pair for later updates
        self.state_history.append((current_state, move, immediate_reward))
        
        # Save current state info for learning from opponent's move
        self.last_state = current_state
        self.last_move = move

        return move

    def update(self, move_other_player):
        """Update the agent's internal state based on opponent's move and learn from it"""
        # Get state before opponent's move
        pre_opponent_state = self.get_state_key()
        pre_move_score = self.calculate_connectivity_score()
        
        # Apply opponent's move
        self.set_hex(self.adv_number, move_other_player)
        
        # Get state after opponent's move
        post_opponent_state = self.get_state_key()
        post_move_score = self.calculate_connectivity_score()
        
        # Calculate reward (negative if opponent improved their position)
        opponent_move_reward = post_move_score - pre_move_score
        
        # Add to experience history for learning
        if self.last_state is not None and self.last_move is not None:
            # Learn from opponent's response to our last move
            self.state_history.append((post_opponent_state, None, opponent_move_reward))
            
            # Also add to appropriate replay buffer - our state now includes opponent's move
            if self.player_number == 1:
                self.replay_buffer_p1.append((self.last_state, self.last_move, opponent_move_reward, post_opponent_state))
            else:
                self.replay_buffer_p2.append((self.last_state, self.last_move, opponent_move_reward, post_opponent_state))
            
            # If opponent's move created a critical position, learn immediately
            if abs(opponent_move_reward) > 3.0:  # Significant change in board evaluation
                self.learn_immediate(self.last_state, self.last_move, opponent_move_reward, post_opponent_state)

    def learn_immediate(self, state, action, reward, next_state):
        """Perform immediate learning for critical positions"""
        current_q = self.get_q_value(state, action)
        
        # Find max Q-value for next state
        next_best_value = -float('inf')
        available_moves = self.free_moves()
        
        if available_moves:
            next_best_value = max([self.get_q_value(next_state, next_action) 
                                   for next_action in available_moves])
        else:
            next_best_value = 0
            
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * next_best_value - current_q)
        self.update_q_value(state, action, new_q)

    def learn(self, final_reward):
        """Update Q-values based on reward and state history"""
        if not self.state_history:
            return

        # Add final reward to the last state-action pair
        if self.state_history:
            # Find the last state with an action (not an opponent move state)
            for i in range(len(self.state_history) - 1, -1, -1):
                if self.state_history[i][1] is not None:  # Has an action
                    self.state_history[i] = (self.state_history[i][0], 
                                            self.state_history[i][1], 
                                            final_reward)
                    break

        # Backward update of Q-values (TD learning)
        for i in range(len(self.state_history) - 1, -1, -1):
            state, action, reward = self.state_history[i]
            
            # Skip opponent move states (no action)
            if action is None:
                continue
                
            # Get current Q-value
            current_q = self.get_q_value(state, action)
            
            if i == len(self.state_history) - 1:  # Final state
                # For terminal state, just update with reward
                new_q = current_q + self.alpha * (reward - current_q)
            else:
                # For non-terminal states, consider next state's best action
                next_state = self.state_history[i+1][0]
                
                # Find best next action's value
                next_best_value = -float('inf')
                temp_board = self.copy()
                available_moves = temp_board.free_moves()
                
                if available_moves:
                    next_best_value = max([self.get_q_value(next_state, next_action)
                                         for next_action in available_moves])
                else:
                    next_best_value = 0
                
                # Q-learning update formula
                new_q = current_q + self.alpha * (reward + self.gamma * next_best_value - current_q)
            
            # Update Q-value
            self.update_q_value(state, action, new_q)
            
        # Experience replay learning
        self.learn_from_replay()
            
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Clear history for next game
        self.state_history = []
        self.moves_counter = 0
        self.last_state = None
        self.last_move = None

    def learn_from_replay(self):
        """Learn from random samples in the experience replay buffer for current player role"""
        # Choose appropriate replay buffer based on player number
        replay_buffer = self.replay_buffer_p1 if self.player_number == 1 else self.replay_buffer_p2
        
        if len(replay_buffer) < self.replay_batch_size:
            return
            
        # Sample random experiences
        samples = random.sample(replay_buffer, min(self.replay_batch_size, len(replay_buffer)))
        
        for state, action, reward, next_state in samples:
            # Skip opponent move states (no action)
            if action is None:
                continue
                
            # Get current Q value
            current_q = self.get_q_value(state, action)
            
            # Create temporary board to find available next moves
            temp_board = self.copy()
            # Apply the current action
            if action is not None:
                temp_board.set_hex(self.player_number, action)
            available_moves = temp_board.free_moves()
            
            # Calculate next best action value
            if available_moves:
                next_best_value = max([self.get_q_value(next_state, next_action)
                                     for next_action in available_moves])
            else:
                next_best_value = 0
                
            # Update Q value
            new_q = current_q + self.alpha * (reward + self.gamma * next_best_value - current_q)
            self.update_q_value(state, action, new_q)

    def save_q_table(self, filename_prefix="enhanced_rl_agent"):
        """Save the Q-tables to separate files"""
        # Save player 1 Q-table
        filename_p1 = f"{filename_prefix}_p1.npy"
        np.save(filename_p1, dict(self.q_table_p1))
        
        # Save player 2 Q-table
        filename_p2 = f"{filename_prefix}_p2.npy"
        np.save(filename_p2, dict(self.q_table_p2))
        
        print(f"Q-tables saved with prefix {filename_prefix}")
        print(f"Player 1 Q-table: {len(self.q_table_p1)} entries")
        print(f"Player 2 Q-table: {len(self.q_table_p2)} entries")

    def load_q_table(self, filename_prefix="enhanced_rl_agent"):
        """Load the Q-tables from separate files"""
        success = True
        
        # Load player 1 Q-table
        filename_p1 = f"{filename_prefix}_p1.npy"
        try:
            loaded_dict_p1 = np.load(filename_p1, allow_pickle=True).item()
            self.q_table_p1 = defaultdict(float, loaded_dict_p1)
            print(f"Loaded Player 1 Q-table from {filename_p1} with {len(self.q_table_p1)} entries")
        except FileNotFoundError:
            print(f"File {filename_p1} not found. Starting with an empty Player 1 Q-table.")
            success = False
        except Exception as e:
            print(f"Error loading Player 1 Q-table: {e}")
            success = False
            
        # Load player 2 Q-table
        filename_p2 = f"{filename_prefix}_p2.npy"
        try:
            loaded_dict_p2 = np.load(filename_p2, allow_pickle=True).item()
            self.q_table_p2 = defaultdict(float, loaded_dict_p2)
            print(f"Loaded Player 2 Q-table from {filename_p2} with {len(self.q_table_p2)} entries")
        except FileNotFoundError:
            print(f"File {filename_p2} not found. Starting with an empty Player 2 Q-table.")
            success = False
        except Exception as e:
            print(f"Error loading Player 2 Q-table: {e}")
            success = False
            
        return success