// Translated from bitboard_experiment.py.
// Implements alpha-beta search for a single piece tour game on an N x N board.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int UNVISITED = 1;
constexpr int VISITED = 0;

enum class Piece { Rook, King, Queen, Knight };

std::string piece_to_string(Piece p) {
	switch (p) {
		case Piece::Rook: return "ROOK";
		case Piece::King: return "KING";
		case Piece::Queen: return "QUEEN";
		case Piece::Knight: return "KNIGHT";
	}
	throw std::runtime_error("Unknown piece");
}

class Board {
public:
	Board(int board_size, Piece piece)
		: board_size_(board_size),
		  cells_(board_size * board_size),
		  piece_(piece),
		  piece_pos_(-1),
		  unvisited_mask_(full_mask(board_size * board_size)) {
		if (cells_ > 64) {
			throw std::invalid_argument("Board too large for 64-bit mask");
		}
		precompute_masks();
	}

	int result(int current_state = 1, int alpha = kNegInf, int beta = kPosInf) {
		const StateKey key{static_cast<uint8_t>(piece_pos_), unvisited_mask_};
		const auto it = memo_.find(key);
		if (it != memo_.end()) return it->second;

		uint64_t legal = move_masks_[piece_pos_] & unvisited_mask_;
		if (legal == 0) {
			const int winner = -current_state;
			memo_.emplace(key, winner);
			return winner;
		}

		bool pruned = false;
		int current_value = (current_state == 1) ? kNegInf : kPosInf;

		while (legal) {
			const int idx = ctz_and_clear(legal);
			const uint64_t bit = 1ULL << idx;
			const int prev_pos = piece_pos_;
			piece_pos_ = idx;
			unvisited_mask_ &= ~bit;
			const int value = result(-current_state, alpha, beta);
			unvisited_mask_ |= bit;
			piece_pos_ = prev_pos;

			if (current_state == 1) {
				current_value = std::max(current_value, value);
				alpha = std::max(alpha, current_value);
			} else {
				current_value = std::min(current_value, value);
				beta = std::min(beta, current_value);
			}

			if (beta <= alpha) {
				pruned = true;
				break;
			}
		}

		if (!pruned) memo_.emplace(key, current_value);
		return current_value;
	}

	std::vector<std::pair<int, int>> winning_initial_positions() {
		memo_.clear();
		unvisited_mask_ = full_mask(cells_);

		std::vector<std::pair<int, int>> winning_cells;
		const int k = (board_size_ + 1) / 2;
		const int total_checks = k * (k + 1) / 2; // number of (r,c) pairs with r<=c<k
		int checked = 0;
		for (int r = 0; r < k; ++r) {
			for (int c = r; c < k; ++c) {
				const int idx = r * board_size_ + c;
				const uint64_t bit = 1ULL << idx;
				piece_pos_ = idx;
				unvisited_mask_ &= ~bit;
				const int res = result(/*current_state=*/1);
				unvisited_mask_ |= bit;
				if (res == 1) add_symmetry_cells(r, c, winning_cells);
				++checked;
				std::cout << "[progress] checked " << checked << "/" << total_checks << " seeds, wins so far: " << winning_cells.size() << "\n";
			}
		}

		std::sort(winning_cells.begin(), winning_cells.end());
		winning_cells.erase(std::unique(winning_cells.begin(), winning_cells.end()), winning_cells.end());
		return winning_cells;
	}

private:
	static constexpr int kNegInf = -1000000000;
	static constexpr int kPosInf = 1000000000;

	struct StateKey {
		uint8_t pos;
		uint64_t mask;
		bool operator==(const StateKey& other) const { return pos == other.pos && mask == other.mask; }
	};

	struct StateKeyHash {
		size_t operator()(const StateKey& k) const noexcept {
			// Simple mix; sufficient because mask carries most entropy.
			return static_cast<size_t>(k.mask ^ (static_cast<uint64_t>(k.pos) << 1));
		}
	};

	int board_size_;
	int cells_;
	Piece piece_;
	int piece_pos_;
	uint64_t unvisited_mask_;
	std::vector<uint64_t> move_masks_;
	std::unordered_map<StateKey, int, StateKeyHash> memo_;

	static uint64_t full_mask(int cells) {
		if (cells >= 64) return ~0ULL;
		return (1ULL << cells) - 1ULL;
	}

	static int ctz_and_clear(uint64_t& x) {
		const int idx = __builtin_ctzll(x);
		x &= x - 1;
		return idx;
	}

	void precompute_masks() {
		move_masks_.assign(cells_, 0ULL);
		for (int r = 0; r < board_size_; ++r) {
			for (int c = 0; c < board_size_; ++c) {
				const int idx = r * board_size_ + c;
				move_masks_[idx] = legal_mask_from(r, c);
			}
		}
	}

	uint64_t legal_mask_from(int r, int c) const {
		const int n = board_size_;
		uint64_t mask = 0ULL;

		switch (piece_) {
			case Piece::Rook: {
				for (int rr = r - 1; rr >= 0; --rr) mask |= (1ULL << (rr * n + c));
				for (int rr = r + 1; rr < n; ++rr) mask |= (1ULL << (rr * n + c));
				for (int cc = c - 1; cc >= 0; --cc) mask |= (1ULL << (r * n + cc));
				for (int cc = c + 1; cc < n; ++cc) mask |= (1ULL << (r * n + cc));
				break;
			}
			case Piece::King: {
				static const int dirs[8][2] = {
					{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
				};
				for (auto d : dirs) {
					const int nr = r + d[0];
					const int nc = c + d[1];
					if (0 <= nr && nr < n && 0 <= nc && nc < n) mask |= (1ULL << (nr * n + nc));
				}
				break;
			}
			case Piece::Queen: {
				for (int rr = r - 1; rr >= 0; --rr) mask |= (1ULL << (rr * n + c));
				for (int rr = r + 1; rr < n; ++rr) mask |= (1ULL << (rr * n + c));
				for (int cc = c - 1; cc >= 0; --cc) mask |= (1ULL << (r * n + cc));
				for (int cc = c + 1; cc < n; ++cc) mask |= (1ULL << (r * n + cc));
				for (int rr = r - 1, cc = c - 1; rr >= 0 && cc >= 0; --rr, --cc) mask |= (1ULL << (rr * n + cc));
				for (int rr = r - 1, cc = c + 1; rr >= 0 && cc < n; --rr, ++cc) mask |= (1ULL << (rr * n + cc));
				for (int rr = r + 1, cc = c - 1; rr < n && cc >= 0; ++rr, --cc) mask |= (1ULL << (rr * n + cc));
				for (int rr = r + 1, cc = c + 1; rr < n && cc < n; ++rr, ++cc) mask |= (1ULL << (rr * n + cc));
				break;
			}
			case Piece::Knight: {
				static const int dirs[8][2] = {
					{2, 1}, {2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}, {-2, 1}, {-2, -1},
				};
				for (auto d : dirs) {
					const int nr = r + d[0];
					const int nc = c + d[1];
					if (0 <= nr && nr < n && 0 <= nc && nc < n) mask |= (1ULL << (nr * n + nc));
				}
				break;
			}
		}
		return mask;
	}

	void add_symmetry_cells(int r, int c, std::vector<std::pair<int, int>>& out) const {
		const int n = board_size_;
		const int r2 = n - 1 - r;
		const int c2 = n - 1 - c;

		out.emplace_back(r, c);
		out.emplace_back(r, c2);
		out.emplace_back(r2, c);
		out.emplace_back(r2, c2);
		out.emplace_back(c, r);
		out.emplace_back(c2, r);
		out.emplace_back(c, r2);
		out.emplace_back(c2, r2);
	}
};

} // namespace

int main() {
	const int board_size = 5;
	const Piece piece = Piece::Queen;

	const auto start = std::chrono::steady_clock::now();
	Board board(board_size, piece);
	const auto winning = board.winning_initial_positions();
	const auto end = std::chrono::steady_clock::now();

	std::cout << "Piece: " << piece_to_string(piece) << " on " << board_size << "x" << board_size << " board\n";
	std::cout << "Winning initial positions (row, col):";
	for (const auto& cell : winning) {
		std::cout << " (" << cell.first << ", " << cell.second << ")";
	}
	std::cout << "\n";

	const auto elapsed = std::chrono::duration<double>(end - start).count();
	std::cout << "Elapsed seconds: " << elapsed << "\n";
	return 0;
}
