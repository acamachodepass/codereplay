// tic_tac_toe.cu
#include "tic_tac_toe.h"

// Board class implementation

Board::Board() {
    reset();
}

void Board::reset() {
    for(int r = 0; r < BOARD_SIZE; r++) {
        for(int c = 0; c < BOARD_SIZE; c++) {
            board[r][c] = CellState::Empty;
        }
    }
    while(!player1Pieces.empty()) player1Pieces.pop();
    while(!player2Pieces.empty()) player2Pieces.pop();
}

bool Board::isCellEmpty(int row, int col) const {
    return board[row][col] == CellState::Empty;
}

bool Board::markCell(int row, int col, CellState player) {
    if(row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return false;
    if(board[row][col] != CellState::Empty) return false;

    board[row][col] = player;
    if(player == CellState::Player1) {
        player1Pieces.push({row, col});
        if(player1Pieces.size() > MAX_PIECES) removeOldestPiece(player);
    } else if(player == CellState::Player2) {
        player2Pieces.push({row, col});
        if(player2Pieces.size() > MAX_PIECES) removeOldestPiece(player);
    }

    return true;
}

void Board::removeOldestPiece(CellState player) {
    if(player == CellState::Player1 && !player1Pieces.empty()) {
        auto oldest = player1Pieces.front();
        player1Pieces.pop();
        board[oldest.first][oldest.second] = CellState::Empty;
    }
    else if(player == CellState::Player2 && !player2Pieces.empty()) {
        auto oldest = player2Pieces.front();
        player2Pieces.pop();
        board[oldest.first][oldest.second] = CellState::Empty;
    }
}

bool Board::validatePieceLimit(CellState player) {
    int count = 0;
    for(int r = 0; r < BOARD_SIZE; r++) {
        for(int c = 0; c < BOARD_SIZE; c++) {
            if(board[r][c] == player) count++;
        }
    }
    return count <= MAX_PIECES;
}

bool Board::checkWin(CellState player) {
    // Horizontal check
    for(int r = 0; r < BOARD_SIZE; r++) {
        int consec = 0;
        for(int c = 0; c < BOARD_SIZE; c++) {
            if(board[r][c] == player) consec++; else consec = 0;
            if(consec == 4) return true;
        }
    }
    // Vertical check
    for(int c = 0; c < BOARD_SIZE; c++) {
        int consec = 0;
        for(int r = 0; r < BOARD_SIZE; r++) {
            if(board[r][c] == player) consec++; else consec = 0;
            if(consec == 4) return true;
        }
    }
    // Diagonal checks
    // Main diagonals and anti-diagonals of length 4 only exist in this 4x4 board
    
    // Main diagonal
    int consec = 0;
    for(int i = 0; i < BOARD_SIZE; i++) {
        if(board[i][i] == player) consec++; else consec=0;
    }
    if(consec == 4) return true;

    // Anti diagonal
    consec = 0;
    for(int i = 0; i < BOARD_SIZE; i++) {
        if(board[i][BOARD_SIZE-1-i] == player) consec++; else consec=0;
    }
    if(consec == 4) return true;

    return false;
}

void Board::print() const {
    // Top border
    std::cout << "-----------------\n";
    for(int r = 0; r < BOARD_SIZE; r++) {
        std::cout << "|";
        for(int c = 0; c < BOARD_SIZE; c++) {
            std::cout << " " << static_cast<char>(board[r][c]) << " |";
        }
        std::cout << "\n-----------------\n";
    }
}

const CellState* Board::data() const {
    return &board[0][0];
}

void Board::getPiecesForPlayer(CellState player, std::vector<int>& rows, std::vector<int>& cols) const {
    rows.clear();
    cols.clear();
    for(int r = 0; r < BOARD_SIZE; r++) {
        for(int c = 0; c < BOARD_SIZE; c++) {
            if(board[r][c] == player) {
                rows.push_back(r);
                cols.push_back(c);
            }
        }
    }
}


// GPU helper device function to get random int index
__device__ int getRandomIndex(curandState* state, int max) {
    return (int)(curand_uniform(state) * max);
}

// Modified GPU kernel with 50% chance first empty cell or random empty cell
__global__ void playerMoveKernel(CellState* d_board, int* d_pRows, int* d_pCols, int numPieces,
                                 CellState player, int* moveRow, int* moveCol, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        curandState localState;
        curand_init(seed, idx, 0, &localState);

        // Collect empty cells
        int emptyIndices[BOARD_SIZE * BOARD_SIZE];
        int emptyCount = 0;
        for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
            if (d_board[i] == CellState::Empty) {
                emptyIndices[emptyCount++] = i;
            }
        }

        if (emptyCount == 0) {
            *moveRow = -1;
            *moveCol = -1;
            return;
        }

        float choice = curand_uniform(&localState);
        int chosenIndex;
        if (choice < 0.5f) {
            // 50% chance pick first empty cell
            chosenIndex = emptyIndices[0];
        } else {
            // 50% chance pick random empty cell
            int randIdx = getRandomIndex(&localState, emptyCount);
            chosenIndex = emptyIndices[randIdx];
        }

        *moveRow = chosenIndex / BOARD_SIZE;
        *moveCol = chosenIndex % BOARD_SIZE;
    }
}

// CUDA error checking utility
#define CUDA_CALL(call) do { cudaError_t err = call; if(err != cudaSuccess) { \
    std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; exit(EXIT_FAILURE); } } while(0)

int main() {
    int nDevices = 0;
    CUDA_CALL(cudaGetDeviceCount(&nDevices));
    if(nDevices == 0) {
        std::cout << "No GPUs detected. Exiting..." << std::endl;
        return 0;
    }

    int devPlayer1 = 0;
    int devPlayer2 = (nDevices > 1) ? 1 : 0;

    Board board;
    board.reset();

    std::cout << "Starting Tic-Tac-Toe 4 in a Row on " << nDevices << " GPU(s)!" << std::endl;

    int totalMoves = 0;
    const int MAX_MOVES = 20;
    unsigned long long seed = static_cast<unsigned long long>(time(NULL));

    CellState currentPlayer = CellState::Player1;

    // Device allocations for board and move communication
    CellState* d_board = nullptr;
    int *d_pRows = nullptr, *d_pCols = nullptr;
    int *d_moveRow = nullptr, *d_moveCol = nullptr;

    CUDA_CALL(cudaMalloc(&d_board, sizeof(CellState)*BOARD_SIZE*BOARD_SIZE));
    CUDA_CALL(cudaMalloc(&d_pRows, sizeof(int)*MAX_PIECES));
    CUDA_CALL(cudaMalloc(&d_pCols, sizeof(int)*MAX_PIECES));
    CUDA_CALL(cudaMalloc(&d_moveRow, sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_moveCol, sizeof(int)));

    int movesLeft1 = MAX_MOVES;
    int movesLeft2 = MAX_MOVES;

    while(totalMoves < MAX_MOVES) {

        CUDA_CALL(cudaMemcpy(d_board, board.data(), sizeof(CellState)*BOARD_SIZE*BOARD_SIZE, cudaMemcpyHostToDevice));
        std::vector<int> pRows, pCols;
        board.getPiecesForPlayer(currentPlayer, pRows, pCols);
        int numPieces = static_cast<int>(pRows.size());
        CUDA_CALL(cudaMemcpy(d_pRows, pRows.data(), sizeof(int)*numPieces, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_pCols, pCols.data(), sizeof(int)*numPieces, cudaMemcpyHostToDevice));

        CUDA_CALL(cudaSetDevice(currentPlayer == CellState::Player1 ? devPlayer1 : devPlayer2));

        playerMoveKernel<<<1,1>>>(d_board, d_pRows, d_pCols, numPieces, currentPlayer, d_moveRow, d_moveCol, seed);
        CUDA_CALL(cudaDeviceSynchronize());

        seed++;

        int moveRow = -1, moveCol = -1;
        CUDA_CALL(cudaMemcpy(&moveRow, d_moveRow, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(&moveCol, d_moveCol, sizeof(int), cudaMemcpyDeviceToHost));

        if(moveRow == -1 || moveCol == -1) {
            std::cout << "No moves left. Game ends in a draw." << std::endl;
            break;
        }

        board.markCell(moveRow, moveCol, currentPlayer);

        if(currentPlayer == CellState::Player1) {
            std::cout << "Move " << totalMoves + 1 << " - Player 1's turn. Remaining moves for Player 1: " 
                      << movesLeft1 << ", Player 2: " << movesLeft2 << std::endl;
            board.print();
            movesLeft1--;
        } else if(currentPlayer == CellState::Player2) {
            std::cout << "Move " << totalMoves + 1 << " - Player 2's turn. Remaining moves for Player 1: " 
                      << movesLeft1 << ", Player 2: " << movesLeft2 << std::endl;
            board.print();
            movesLeft2--;
        }

        if(board.checkWin(currentPlayer)) {
            board.print();
            std::cout << "Player " << (currentPlayer == CellState::Player1 ? '1' : '2') << " wins! Game over." << std::endl;
            break;
        }

        if(currentPlayer == CellState::Player2) {
            totalMoves++;
        } 

        if(totalMoves == MAX_MOVES) {
            //board.print();
            std::cout << "Move limit reached. Game ends in a draw." << std::endl;
            std::cout.flush();
            break;
        }

        currentPlayer = (currentPlayer == CellState::Player1) ? CellState::Player2 : CellState::Player1;
    }

    CUDA_CALL(cudaFree(d_board));
    CUDA_CALL(cudaFree(d_pRows));
    CUDA_CALL(cudaFree(d_pCols));
    CUDA_CALL(cudaFree(d_moveRow));
    CUDA_CALL(cudaFree(d_moveCol));

    return 0;
}

