#include <bits/stdc++.h>
#include <random>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro_io **/

/* tuple */
// out
namespace aux {
    template<typename T, unsigned N, unsigned L>
    struct tp {
        static void output(std::ostream& os, const T& v) {
            os << std::get<N>(v) << ", ";
            tp<T, N + 1, L>::output(os, v);
        }
    };
    template<typename T, unsigned N>
    struct tp<T, N, N> {
        static void output(std::ostream& os, const T& v) { os << std::get<N>(v); }
    };
}
template<typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) {
    os << '[';
    aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t);
    return os << ']';
}

template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x);

/* pair */
// out
template<class S, class T>
std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) {
    return os << "[" << p.first << ", " << p.second << "]";
}
// in
template<class S, class T>
std::istream& operator>>(std::istream& is, const std::pair<S, T>& p) {
    return is >> p.first >> p.second;
}

/* container */
// out
template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) {
    bool f = true;
    os << "[";
    for (auto& y : x) {
        os << (f ? "" : ", ") << y;
        f = false;
    }
    return os << "]";
}
// in
template <
    class T,
    class = decltype(std::begin(std::declval<T&>())),
    class = typename std::enable_if<!std::is_same<T, std::string>::value>::type
>
std::istream& operator>>(std::istream& is, T& a) {
    for (auto& x : a) is >> x;
    return is;
}

/* struct */
template<typename T>
auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) {
    out << t.stringify();
    return out;
}

/* setup */
struct IOSetup {
    IOSetup(bool f) {
        if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); }
        std::cout << std::fixed << std::setprecision(15);
    }
} iosetup(true);

/** string formatter **/
template<typename... Ts>
std::string format(const std::string& f, Ts... t) {
    size_t l = std::snprintf(nullptr, 0, f.c_str(), t...);
    std::vector<char> b(l + 1);
    std::snprintf(&b[0], l + 1, f.c_str(), t...);
    return std::string(&b[0], &b[0] + l);
}

template<typename T>
std::string stringify(const T& x) {
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

/* dump */
#define ENABLE_DUMP
#ifdef ENABLE_DUMP
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
#else
#define dump(...) void(0);
#endif

/* timer */
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() { return (time() - t - paused) * 1000.0; }
} timer;

/* rand */
struct Xorshift {
    uint64_t x = 88172645463325252LL;
    void set_seed(unsigned seed, int rep = 100) { x = uint64_t((seed + 1) * 10007); for (int i = 0; i < rep; i++) next_int(); }
    unsigned next_int() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    unsigned next_int(unsigned mod) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % mod; }
    unsigned next_int(unsigned l, unsigned r) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % (r - l + 1) + l; } // inclusive
    double next_double() { return double(next_int()) / UINT_MAX; }
} rnd;

/* shuffle */
template<typename T>
void shuffle_vector(std::vector<T>& v, Xorshift& rnd) {
    int n = v.size();
    for (int i = n - 1; i >= 1; i--) {
        int r = rnd.next_int(i);
        std::swap(v[i], v[r]);
    }
}

/* split */
std::vector<std::string> split(std::string str, const std::string& delim) {
    for (char& c : str) if (delim.find(c) != std::string::npos) c = ' ';
    std::istringstream iss(str);
    std::vector<std::string> parsed;
    std::string buf;
    while (iss >> buf) parsed.push_back(buf);
    return parsed;
}

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}

template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

using std::vector, std::string;
using std::cin, std::cout, std::cerr, std::endl;



constexpr int N = 30;
constexpr int MAX_TURN = 300;
constexpr int dy[] = { 0, -1, 0, 1 };
constexpr int dx[] = { 1, 0, -1, 0 };
constexpr char d2C[5] = "RULD";
constexpr char d2c[5] = "ruld";
int c2d[256];
constexpr char WALL = '#';
constexpr char EMPTY = '.';
constexpr char HUMAN = 'H';

struct Pet {
    int y, x, t;
    Pet(int y = -1, int x = -1, int t = -1) : y(y), x(x), t(t) {}
};

struct Human {
    int y, x;
    Human(int y = -1, int x = -1) : y(y), x(x) {}
};

struct State {

    static constexpr bool debug = false;

    std::istream& in;
    std::ostream& out;

    int num_pets;
    vector<Pet> pets;

    int num_humans;
    vector<Human> humans;

    char board[N][N];
    int pet_count[N][N];

    State(std::istream& in, std::ostream& out) : in(in), out(out) { init(); }

    void print_board() const {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                char np = (pet_count[y][x] == 0) ? '.' : char(std::min(9, pet_count[y][x]) + '0');
                cerr << format("%c%c|", board[y][x], np);
            }
            cerr << '\n';
        }
    }

    void init() {
        if (debug) cerr << format("--- %s called ---\n", __FUNCTION__);
        Fill(board, '.');
        Fill(pet_count, 0);
        in >> num_pets;
        if (debug) cerr << format("num_pets: %d\n", num_pets);
        for (int pid = 0; pid < num_pets; pid++) {
            int y, x, t;
            cin >> y >> x >> t;
            x--; y--;
            if (debug) cerr << format("position of pet %d: (%d, %d)\n", pid, y, x);
            pets.emplace_back(y, x, t);
            pet_count[y][x]++;
        }
        in >> num_humans;
        if (debug) cerr << format("num_humans: %d\n", num_humans);
        for (int hid = 0; hid < num_humans; hid++) {
            int y, x;
            cin >> y >> x;
            x--; y--;
            if (debug) cerr << format("position of human %d: (%d, %d)\n", hid, y, x);
            humans.emplace_back(y, x);
            board[y][x] = HUMAN;
        }
        if (debug) print_board(); 
        if (debug) cerr << format("--- %s end ---\n", __FUNCTION__);
    }

    vector<string> load() {
        vector<string> pet_moves(num_pets);
        cin >> pet_moves;
        for (int pid = 0; pid < num_pets; pid++) {
            auto& pet = pets[pid];
            for (char c : pet_moves[pid]) {
                pet_count[pet.y][pet.x]--;
                int d = c2d[c];
                pet.y += dy[d];
                pet.x += dx[d];
                pet_count[pet.y][pet.x]++;
            }
        }
        return pet_moves;
    }

    inline bool is_inside(int y, int x) const {
        return 0 <= y && y < N && 0 <= x && x < N;
    }

    bool can_place(int y, int x) const {
        if (!is_inside(y, x) || board[y][x] != EMPTY || pet_count[y][x]) return false;
        for (int d = 0; d < 4; d++) {
            int ny = y + dy[d], nx = x + dx[d];
            if (!is_inside(ny, nx)) continue;
            if (pet_count[ny][nx]) return false;
        }
        return true;
    }

    string calc_moves() const {
        string moves(num_humans, '.');
        for (int hid = 0; hid < num_humans; hid++) {
            auto [y, x] = humans[hid];
            for (int d = 0; d < 4; d++) {
                if (can_place(y + dy[d], x + dx[d])) {
                    moves[hid] = d2c[d];
                    break;
                }
            }
        }
        return moves;
    }

    void do_moves(const string& moves) {
        for (int hid = 0; hid < num_humans; hid++) {
            auto& [y, x] = humans[hid];
            char c = moves[hid];
            if (c == '.') continue;
            if (isupper(c)) {
                y += dy[c2d[c]];
                x += dx[c2d[c]];
            }
            if (islower(c)) {
                board[y + dy[c2d[c]]][x + dx[c2d[c]]] = WALL;
            }
        }
    }

    void solve() {
        for (int turn = 0; turn < MAX_TURN; turn++) {
            if (debug) {
                cerr << format("--- turn %d ---\n", turn);
                print_board();
            }
            auto moves = calc_moves();
            if (debug) cerr << format("move %3d: %s\n", turn, moves.c_str());
            do_moves(moves);
            cout << moves << endl;
            load();
        }
    }

};

int main() {

    c2d['R'] = c2d['r'] = 0;
    c2d['U'] = c2d['u'] = 1;
    c2d['L'] = c2d['l'] = 2;
    c2d['D'] = c2d['d'] = 3;

    State state(cin, cout);

    state.solve();

    return 0;
}