from collections import deque
import time
from hexboard import HexBoard
from base_player import Player

TIME_LIMIT = 3.0

class MarBys_Player(Player):
    """Jugador siguiendo algoritmo minimax con poda alpha-beta"""
    def __init__(self, player_id, depth=2):
        super().__init__(player_id)
        self.depth = depth
        self.heuristic = self.default_heuristic

    def play(self, board: HexBoard) -> tuple:
        """Elige el mejor movimiento entre los disponibles"""
        
        start_time = time.time()
        self.start_time = start_time

        _,best_move = self.minimax(board, self.depth, True, float("-inf"), float("inf"))

        duration = time.time() - start_time
        print(f"Tiempo total de jugada: {duration:.3f}s")
        return best_move

    def minimax(self, board: HexBoard, depht, maximizing, alpha, beta) -> tuple:
        """Algoritmo minimax con poda alpha-beta"""
        otherPlayer_id = 1 if self.player_id == 2 else 2

        if time.time() - self.start_time > TIME_LIMIT:
            return self.heuristic(board, self.player_id), None
        if depht == 0:
            return self.heuristic(board, self.player_id), None
        if board.check_connection(self.player_id):
            return float("inf"), None
        if board.check_connection(otherPlayer_id):
            return float("-inf"), None
        
        best_move = None
        board_copy = board.clone()
        best_score = float("-inf") if maximizing else float("inf")
        possible_moves = board.get_possible_moves()

        for move in possible_moves:
            board_copy.place_piece(*move, self.player_id) if maximizing else board_copy.place_piece(*move, otherPlayer_id)
            maximizing2 = False if maximizing else True
            score,_ = self.minimax(board_copy, depht-1, maximizing2, float("-inf"), float("inf"))
            board_copy.board[move[0]][move[1]] = 0

            if maximizing: 
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            if alpha >= beta: break

        if best_move is None and possible_moves:
            return possible_moves[0] #implementar estrategia para que el jugador salga del paso

        return best_score,best_move

    def default_heuristic(self, board:HexBoard, player_id: int) -> float:
        """Heurística súper basica que cuenta cuantas fichas el adversario y el jugador tiene conectadas"""
        otherPlayer_id = 1 if player_id == 2 else 2

        count1 = self.countConnected(board, player_id)
        count2 = self.countConnected(board, otherPlayer_id)

        return (count2 - count1)/max(count1, count2) if max(count1, count2) != 0 else 0
    
    def countConnected(self, board: HexBoard, player_id: int):
        """Devuelve cuántas piezas están conectadas"""
        counted = set()
        connected = 0
        for row in range(board.size):
            for col in range(board.size):
                if board.board[row][col] == player_id:
                    neighbors = self.getNeighbors(board, row, col)
                    for piece in neighbors:
                        if board.board[piece[0]][piece[1]] == player_id and piece not in counted:
                            counted.add(piece)
                            connected+=1
        return connected
    
    def getNeighbors(self, board: HexBoard, row: int, col: int):
        """Devuelve las fichas adyacentes a la actual"""
        result = []
        directions = [
            (0,-1), # Izquierda
            (0,1), # Derecha
            (-1,0), # Arriba
            (1,0), # Abajo
            (-1,1), # Arriba derecha
            (1,-1) # Abajo izquierda
        ]
        for neighbor in directions:
            nrow = row + neighbor[0]
            ncol = col + neighbor[1]
            if (nrow >= 0 | nrow < board.size) & (ncol >= 0 | ncol < board.size):
                result.append((nrow, ncol))
        return result
    