from core.grid import Grid

class Controller:
    def __init__(self, size, player1, player2):
        self._grid = Grid(size)
        self._player1 = player1
        self._player2 = player2
        self._current_player = 1
        self._winner = 0
        self._move_count = 0
        self._size = size

    def update(self):
        if self._winner != 0 or self._move_count >= self._size * self._size:
            return  # Game is already over

        if self._current_player == 1:
            coordinates = self._player1.step()
            self._grid.set_hex(self._current_player, coordinates)
            self._player2.update(coordinates)
            self._sync_agent_grid(self._player2)
            self._current_player = 2
        else:
            coordinates = self._player2.step()
            self._grid.set_hex(self._current_player, coordinates)
            self._player1.update(coordinates)
            self._sync_agent_grid(self._player1)
            self._current_player = 1

        self._move_count += 1
        self._check_win()

    def _check_win(self):
        if self._grid.check_win(1):
            self._winner = 1
        elif self._grid.check_win(2):
            self._winner = 2
        elif self._move_count >= self._size * self._size:
            self._winner = -1  # Draw

    def _sync_agent_grid(self, agent):
        """Synchronize the agent's internal grid with the controller's grid"""
        for x in range(agent.size):
            for y in range(agent.size):
                agent._grid[y][x] = self._grid.get_hex([x, y])
