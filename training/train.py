import argparse
import numpy as np
import random
import time
import os
from agent1 import CustomPlayer
from agent2 import MinimaxPlayer
from board_reading_agent import BoardReadingAgent
from strategic_sacrifice import StrategicSacrificeAgent
from edge_control_agent import EdgeControlAgent
from BridgeAgent import BridgeAgent
from TemplatePattern import TemplatePatternAgent
from ThreatCreation import ThreatCreationAgent
from virtual_connect import VirtualConnectionAgent
from DefensivePosition import DefensivePositioningAgent
from opening_stratergy import OpeningStrategyAgent
from CenterDominance import CenterDominanceAgent
from controller import Controller
from grid import Grid

def train_against_specific_opponents(board_size=5, 
                                    games_per_opponent=5000, 
                                    save_interval=1000, 
                                    load_path=None,
                                    results_file="training_results.txt"):
    """
    Train RL agent specifically against each opponent with equal games as player 1 and player 2
    
    Parameters:
    - board_size: Size of the Hex board
    - games_per_opponent: Number of games to play as each player (P1/P2) against each opponent
    - save_interval: How often to save the model during training
    - load_path: Path to load existing Q-table from
    """
    
    # Create our RL agent with separate Q-tables for player 1 and player 2 roles
    rl_agent = CustomPlayer(board_size, 1, 2, 
                            alpha=0.2,      # Learning rate
                            gamma=0.95,     # Discount factor
                            epsilon=0.5,    # Initial exploration rate
                            load_q_table_path=load_path)
    
    # Create output directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Define all opponent agents
    opponents = [
        {"name": "Minimax", "class": MinimaxPlayer},
        {"name": "BoardReading", "class": BoardReadingAgent},
        {"name": "StrategicSacrifice", "class": StrategicSacrificeAgent},
        {"name": "EdgeControl", "class": EdgeControlAgent},
        {"name": "Bridge", "class": BridgeAgent},
        {"name": "TemplatePattern", "class": TemplatePatternAgent},
        {"name": "ThreatCreation", "class": ThreatCreationAgent},
        {"name": "VirtualConnection", "class": VirtualConnectionAgent},
        {"name": "DefensivePositioning", "class": DefensivePositioningAgent},
        {"name": "OpeningStrategy", "class": OpeningStrategyAgent},
        {"name": "CenterDominance", "class": CenterDominanceAgent}
    ]
    
    print(f"Starting training against {len(opponents)} agents")
    print(f"Each agent: {games_per_opponent} games as P1 + {games_per_opponent} games as P2")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Alpha: {rl_agent.alpha}, Gamma: {rl_agent.gamma}, Initial Epsilon: {rl_agent.epsilon}")
    
    # Open results file
    with open(results_file, 'w') as results:
        results.write(f"Training results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        results.write(f"Board size: {board_size}x{board_size}\n")
        results.write(f"Games per opponent per role: {games_per_opponent}\n\n")
        
        # Track overall stats
        total_games = 0
        total_wins_as_p1 = 0
        total_wins_as_p2 = 0
        total_losses_as_p1 = 0
        total_losses_as_p2 = 0
        total_draws = 0
        
        # Train against each opponent in turn
        for opponent_info in opponents:
            opponent_name = opponent_info["name"]
            opponent_class = opponent_info["class"]
            
            print(f"\n{'='*50}")
            print(f"Training against {opponent_name}")
            print(f"{'='*50}")
            results.write(f"Opponent: {opponent_name}\n")
            
            # Initialize counters for this opponent
            wins_as_p1 = 0
            wins_as_p2 = 0
            losses_as_p1 = 0
            losses_as_p2 = 0
            draws = 0
            
            start_time = time.time()
            
            # PART 1: Play as Player 1 against this opponent
            for game in range(1, games_per_opponent + 1):
                # Create fresh board for this game
                board = Grid(board_size)
                
                # Create player instances
                player1 = CustomPlayer(board_size, 1, 2, 
                                      alpha=rl_agent.alpha,
                                      gamma=rl_agent.gamma, 
                                      epsilon=rl_agent.epsilon)
                player2 = opponent_class(board_size, 2, 1)
                
                # Copy Q-tables from main agent
                player1.q_table_p1 = rl_agent.q_table_p1.copy()
                player1.q_table_p2 = rl_agent.q_table_p2.copy()
                
                # Create controller and run the game
                controller = Controller(board_size, player1, player2)
                
                moves_count = 0
                while controller._winner == 0 and moves_count < board_size * board_size:
                    controller.update()
                    moves_count += 1
                
                # Determine the winner and update stats
                winner = controller._winner
                
                if winner == 1:
                    wins_as_p1 += 1
                    player1.learn(2.0)  # Big positive reward for winning
                elif winner == 2:
                    losses_as_p1 += 1
                    player1.learn(-1.0)  # Negative reward for losing
                else:
                    draws += 1
                    player1.learn(-0.1)  # Small negative reward for draw
                
                # Update main agent's Q-tables with what this instance learned
                for state, actions in player1.q_table_p1.items():
                    rl_agent.q_table_p1[state] = actions
                
                for state, actions in player1.q_table_p2.items():
                    rl_agent.q_table_p2[state] = actions
                
                # Decay epsilon for next game
                rl_agent.epsilon = max(rl_agent.epsilon_min, rl_agent.epsilon * rl_agent.epsilon_decay)
                
                # Save checkpoint and report progress
                if game % save_interval == 0:
                    checkpoint_path = f"models/hex_rl_vs_{opponent_name}_p1_{game}"
                    rl_agent.save_q_table(checkpoint_path)
                    
                    elapsed = time.time() - start_time
                    print(f"P1 vs {opponent_name}: Game {game}/{games_per_opponent} - " 
                          f"Wins: {wins_as_p1} ({wins_as_p1/game*100:.1f}%), " 
                          f"Losses: {losses_as_p1} ({losses_as_p1/game*100:.1f}%), "
                          f"Draws: {draws}, "
                          f"Epsilon: {rl_agent.epsilon:.4f}, "
                          f"Time: {elapsed:.1f}s")
            
            # PART 2: Play as Player 2 against this opponent
            draws_before_p2 = draws  # Track draws before starting P2 games
            
            for game in range(1, games_per_opponent + 1):
                # Create fresh board for this game
                board = Grid(board_size)
                
                # Create player instances
                player1 = opponent_class(board_size, 1, 2)
                player2 = CustomPlayer(board_size, 2, 1, 
                                      alpha=rl_agent.alpha,
                                      gamma=rl_agent.gamma, 
                                      epsilon=rl_agent.epsilon)
                
                # Copy Q-tables from main agent
                player2.q_table_p1 = rl_agent.q_table_p1.copy()
                player2.q_table_p2 = rl_agent.q_table_p2.copy()
                
                # Create controller and run the game
                controller = Controller(board_size, player1, player2)
                
                moves_count = 0
                while controller._winner == 0 and moves_count < board_size * board_size:
                    controller.update()
                    moves_count += 1
                
                # Determine the winner and update stats
                winner = controller._winner
                
                if winner == 2:
                    wins_as_p2 += 1
                    player2.learn(2.0)  # Big positive reward for winning
                elif winner == 1:
                    losses_as_p2 += 1
                    player2.learn(-1.0)  # Negative reward for losing
                else:
                    draws += 1
                    player2.learn(-0.1)  # Small negative reward for draw
                
                # Update main agent's Q-tables with what this instance learned
                for state, actions in player2.q_table_p1.items():
                    rl_agent.q_table_p1[state] = actions
                
                for state, actions in player2.q_table_p2.items():
                    rl_agent.q_table_p2[state] = actions
                
                # Decay epsilon for next game
                rl_agent.epsilon = max(rl_agent.epsilon_min, rl_agent.epsilon * rl_agent.epsilon_decay)
                
                # Save checkpoint and report progress
                if game % save_interval == 0:
                    checkpoint_path = f"/content/drive/MyDrive/AI-rena/hex_rl_vs_{opponent_name}_p2_{game}"
                    rl_agent.save_q_table(checkpoint_path)
                    
                    elapsed = time.time() - start_time
                    print(f"P2 vs {opponent_name}: Game {game}/{games_per_opponent} - " 
                          f"Wins: {wins_as_p2} ({wins_as_p2/game*100:.1f}%), " 
                          f"Losses: {losses_as_p2} ({losses_as_p2/game*100:.1f}%), "
                          f"Draws: {draws - draws_before_p2}, "
                          f"Epsilon: {rl_agent.epsilon:.4f}, "
                          f"Time: {elapsed:.1f}s")
            
            # Calculate summary stats for this opponent
            total_games_this_opponent = games_per_opponent * 2
            draws_as_p2 = draws - draws_before_p2
            
            win_rate_p1 = wins_as_p1 / games_per_opponent * 100
            win_rate_p2 = wins_as_p2 / games_per_opponent * 100
            overall_win_rate = (wins_as_p1 + wins_as_p2) / total_games_this_opponent * 100
            
            # Save final model for this opponent
            final_path = f"/content/drive/MyDrive/AI-rena//hex_rl_vs_{opponent_name}_final"
            rl_agent.save_q_table(final_path)
            
            # Update overall stats
            total_games += total_games_this_opponent
            total_wins_as_p1 += wins_as_p1
            total_wins_as_p2 += wins_as_p2
            total_losses_as_p1 += losses_as_p1
            total_losses_as_p2 += losses_as_p2
            total_draws += draws
            
            # Write results for this opponent
            elapsed_time = time.time() - start_time
            results.write(f"Games: {total_games_this_opponent}, Time: {elapsed_time:.1f}s\n")
            results.write(f"Win rate as P1: {win_rate_p1:.1f}% ({wins_as_p1}/{games_per_opponent})\n")
            results.write(f"Win rate as P2: {win_rate_p2:.1f}% ({wins_as_p2}/{games_per_opponent})\n")
            results.write(f"Overall win rate: {overall_win_rate:.1f}%\n")
            results.write(f"Draws: {draws} ({draws/total_games_this_opponent*100:.1f}%)\n\n")
            
            print(f"\nCompleted training against {opponent_name}!")
            print(f"Win rate as P1: {win_rate_p1:.1f}%, Win rate as P2: {win_rate_p2:.1f}%")
            print(f"Overall win rate: {overall_win_rate:.1f}%")
            print(f"Total time: {elapsed_time:.1f}s")
        
        # Calculate and write overall stats
        total_win_rate = (total_wins_as_p1 + total_wins_as_p2) / total_games * 100
        total_p1_win_rate = total_wins_as_p1 / (games_per_opponent * len(opponents)) * 100
        total_p2_win_rate = total_wins_as_p2 / (games_per_opponent * len(opponents)) * 100
        
        results.write(f"{'='*50}\n")
        results.write(f"OVERALL TRAINING SUMMARY\n")
        results.write(f"{'='*50}\n")
        results.write(f"Total games: {total_games}\n")
        results.write(f"Overall win rate: {total_win_rate:.1f}%\n")
        results.write(f"Win rate as P1: {total_p1_win_rate:.1f}%\n")
        results.write(f"Win rate as P2: {total_p2_win_rate:.1f}%\n")
        results.write(f"Draws: {total_draws} ({total_draws/total_games*100:.1f}%)\n")
        
    # Save final model after all training
    final_path = f"/content/drive/MyDrive/AI-rena//hex_rl_agent_{board_size}x{board_size}_all_opponents_final"
    rl_agent.save_q_table(final_path)
    print(f"Training complete! Final model saved to {final_path}")
    
    return rl_agent


def main():
    parser = argparse.ArgumentParser(description="Train RL agent against all opponent agents")
    parser.add_argument("--board-size", type=int, default=5, help="Size of the game board")
    parser.add_argument("--games-per-opponent", type=int, default=5000, 
                        help="Number of games to play as each player (P1/P2) against each opponent")
    parser.add_argument("--save-interval", type=int, default=1000, 
                        help="Save checkpoints every N games")
    parser.add_argument("--load-path", type=str, default=None, 
                        help="Path to load existing Q-table from (no extension)")
    parser.add_argument("--results-file", type=str, default="training_results.txt",
                        help="File to save training results")
    
    args = parser.parse_args()
    
    # Start training
    train_against_specific_opponents(
        board_size=args.board_size,
        games_per_opponent=args.games_per_opponent,
        save_interval=args.save_interval,
        load_path=args.load_path,
        results_file=args.results_file
    )

if __name__ == "__main__":
    main()