// tic_tac_toe.h
#ifndef TIC_TAC_TOE_H
#define TIC_TAC_TOE_H

#include <iostream>
#include <vector>
#include <queue>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>


// Constants
constexpr int BOARD_SIZE = 4;
constexpr int MAX_PIECES = 4;

// Enum for cell state
enum class CellState : char {
    Empty = ' ',
    Player1 = 'x',
    Player2 = 'o'
};

class Board {
private:
    CellState board[BOARD_SIZE][BOARD_SIZE];

    // Queues to track the order of pieces for cycling per player
    std::queue<std::pair<int,int>> player1Pieces;
    std::queue<std::pair<int,int>> player2Pieces;

public:
    Board();
    void reset();
    bool markCell(int row, int col, CellState player);
    bool validatePieceLimit(CellState player);
    bool checkWin(CellState player);
    void print() const;
    bool isCellEmpty(int row, int col) const;
    void removeOldestPiece(CellState player);
    const CellState* data() const;       // For GPU transfer: pointer to board data
    void getPiecesForPlayer(CellState player, std::vector<int>& rows, std::vector<int>& cols) const; // For move history
};

__global__ void playerMoveKernel(CellState* d_board, int* d_pRows, int* d_pCols, int numPieces,
                                 CellState player, int* moveRow, int* moveCol, unsigned long long seed);

__device__ int getRandomIndex(curandState* state, int max);

#endif // TIC_TAC_TOE_H
