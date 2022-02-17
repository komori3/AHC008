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
#ifdef _MSC_VER
//#define ENABLE_DUMP
#endif
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

template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

using std::vector, std::string;
using std::cin, std::cout, std::cerr, std::endl;



struct UnionFind {
    vector<int> data;

    UnionFind() = default;

    explicit UnionFind(size_t sz) : data(sz, -1) {}

    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (data[x] > data[y]) std::swap(x, y);
        data[x] += data[y];
        data[y] = x;
        return true;
    }

    int find(int k) {
        if (data[k] < 0) return (k);
        return data[k] = find(data[k]);
    }

    int size(int k) {
        return -data[find(k)];
    }

    bool same(int x, int y) {
        return find(x) == find(y);
    }

    vector<vector<int>> groups() {
        int n = (int)data.size();
        vector< vector< int > > ret(n);
        for (int i = 0; i < n; i++) {
            ret[find(i)].emplace_back(i);
        }
        ret.erase(remove_if(begin(ret), end(ret), [&](const vector< int >& v) {
            return v.empty();
            }), ret.end());
        return ret;
    }
};


constexpr int MAX_TURN = 300;
constexpr char d2C[5] = "RULD";
constexpr char d2c[5] = "ruld";
int c2d[256];
constexpr int inf = INT_MAX / 8;

namespace NSolver {

    constexpr int N = 30;
    constexpr int dy[] = { 0, -1, 0, 1 };
    constexpr int dx[] = { 1, 0, -1, 0 };

    struct Point {
        int y, x;
        Point(int y = -1, int x = -1) : y(y), x(x) {}
        Point operator-() const { return Point(-y, -x); }
        Point& operator+=(const Point& p) { y += p.y; x += p.x; return *this; }
        Point& operator-=(const Point& p) { y -= p.y; x -= p.x; return *this; }
        Point& operator*=(int k) { y *= k; x *= k; return *this; }
        string stringify() const {
            return format("{%d,%d}", y, x);
        }
    };
    Point operator+(const Point& p1, const Point& p2) { return Point(p1) += p2; }
    Point operator-(const Point& p1, const Point& p2) { return Point(p1) -= p2; }
    Point operator*(int k, const Point& p1) { return Point(p1) *= k; }
    const Point dir[4] = { {0, 1}, {-1, 0}, {0, -1}, {1, 0} };

    struct Action {
        enum struct Type {
            MOVE, BLOCK, WAIT
        };
        Type type;
        int y, x, mask, time;
        Action(Type type, int y, int x, int mask, int time) : type(type), y(y), x(x), mask(mask), time(time) {}
        static Action move(int y, int x) {
            return Action(Type::MOVE, y, x, -1, -1);
        }
        static Action move(const Point& p) {
            return move(p.y, p.x);
        }
        static Action block(int mask) {
            return Action(Type::BLOCK, -1, -1, mask, -1);
        }
        static Action wait(int time) {
            return Action(Type::WAIT, -1, -1, -1, time);
        }
        string stringify() const {
            switch (type)
            {
            case Action::Type::MOVE:
                return format("move(%d,%d)", y, x);
            case Action::Type::BLOCK:
                return format("block(%s)", std::bitset<4>(mask).to_string().c_str());
            case Action::Type::WAIT:
                return format("wait(%d)", time);
            default:
                return "";
            }
            return "";
        }
    };

    struct Pet {
        int id, y, x, t;
        Pet(int id = -1, int y = -1, int x = -1, int t = -1) : id(id), y(y), x(x), t(t) {}
        Point get_coord() const { return { y, x }; }
    };

    struct Human {
        int id, y, x;
        std::deque<Action> action_queue;
        Human(int id = -1, int y = -1, int x = -1) : id(id), y(y), x(x) {}
        Point get_coord() const { return { y, x }; }
    };

    struct Task {
        int id;
        int est_cost; // 全ての行動を終えるまでの予定時間
        vector<Action> actions; // 行動列
        bool taken = false;
    };

    struct TrapTask {
        int id;
        // 0~N-4: pet, N-2: block, N-3: human
        // 建設終了チェック
        // -4->-3
        // -3->-2
        vector<Point> area;
        int taken = -1; // human id
    };

    struct State {

        std::istream& in;
        std::ostream& out;

        int turn;

        vector<Pet> pets;
        vector<Human> humans;

        bool blocked[N][N];
        int human_count[N][N];
        int pet_count[N][N];

        bool blocked_tmp[N][N]; // ターン t での柵の設置予定場所に人間が飛び込むのを禁止する
        int human_count_tmp[N][N]; // ターン t での人間の移動予定場所に柵を突き立てるのを禁止する
        // TODO: 人間は移動によって重なることが許されていそうなので、許容する

        State(std::istream& in, std::ostream& out) : in(in), out(out) { init(); }

        void init() {
            Fill(blocked, false);
            Fill(human_count, 0);
            Fill(pet_count, 0);
            turn = 0;
            int num_pets; in >> num_pets;
            for (int pid = 0; pid < num_pets; pid++) {
                int y, x, t;
                cin >> y >> x >> t;
                x--; y--; t--;
                pets.emplace_back(pid, y, x, t);
                pet_count[y][x]++;
            }
            int num_humans; in >> num_humans;
            for (int hid = 0; hid < num_humans; hid++) {
                int y, x;
                cin >> y >> x;
                x--; y--;
                humans.emplace_back(hid, y, x);
                human_count[y][x]++;
            }
        }

        vector<string> load() {
            vector<string> pet_moves(pets.size());
            cin >> pet_moves;
            for (int pid = 0; pid < pets.size(); pid++) {
                auto& pet = pets[pid];
                for (char c : pet_moves[pid]) {
                    pet_count[pet.y][pet.x]--;
                    int d = c2d[c];
                    //dump(turn, pid, d, pet.y, pet.x, pet.y + dy[d], pet.x + dx[d]);
                    pet.y += dy[d];
                    pet.x += dx[d];
                    pet_count[pet.y][pet.x]++;
                }
            }
            return pet_moves;
        }

        inline bool is_inside(int y, int x) const { return 0 <= y && y < N && 0 <= x && x < N; }
        inline bool is_inside(const Point& p) const { return is_inside(p.y, p.x); }

        // (y, x) に柵を設置可能か？
        bool can_place(int y, int x) const {
            // 領域内 || 重複して設置しない || 人間に刺さない || ペットに刺さない
            if (!is_inside(y, x) || blocked_tmp[y][x] || human_count_tmp[y][x] || pet_count[y][x]) return false;
            for (int d = 0; d < 4; d++) {
                int ny = y + dy[d], nx = x + dx[d];
                if (!is_inside(ny, nx)) continue;
                // ペットの 4 近傍に置かない
                if (pet_count[ny][nx]) return false;
            }
            return true;
        }

        // 人間 hid が (gy, gx) に最短経路で移動するための移動方向（候補複数ならランダム）
        char calc_move(int hid, int gy, int gx) {
            static int dist[N][N];

            auto [sy, sx] = humans[hid].get_coord();
            assert(sy != gy || sx != gx);

            // pet, human は無視して目的地からの距離を計算
            Fill(dist, inf);
            std::queue<Point> qu;
            qu.emplace(gy, gx);
            dist[gy][gx] = 0;
            while (!qu.empty()) {
                auto [y, x] = qu.front(); qu.pop();
                for (int d = 0; d < 4; d++) {
                    int ny = y + dy[d], nx = x + dx[d];
                    // 設置予定の柵も含む
                    if (!is_inside(ny, nx) || blocked_tmp[ny][nx] || dist[ny][nx] != inf) continue;
                    dist[ny][nx] = dist[y][x] + 1;
                    qu.emplace(ny, nx);
                }
            }

            if (dist[sy][sx] == inf) return '.'; // 移動不可
            
            // 最短経路移動方向の候補を調べる
            int min_dist = inf;
            vector<int> cands;
            for (int d = 0; d < 4; d++) {
                int ny = sy + dy[d], nx = sx + dx[d];
                // 柵とペットを踏まない
                // 人間については、元からいた場所は避ける (移動先が重複するのは許す)
                if (!is_inside(ny, nx) || blocked_tmp[ny][nx] || human_count[ny][nx] || pet_count[ny][nx] || dist[ny][nx] > min_dist) continue;
                if (dist[ny][nx] < min_dist) {
                    min_dist = dist[ny][nx];
                    cands.clear();
                }
                cands.push_back(d);
            }

            if (min_dist == inf) return '.'; // 移動不可 (pet に包囲されるケースが稀にある)

            // ランダムで選ぶ
            int d = cands[rnd.next_int(cands.size())];
            return d2C[d];
        }

        // 柵の設置
        char calc_block(int hid, int mask) {
            auto [y, x] = humans[hid].get_coord();
            for (int d = 0; d < 4; d++) if (mask >> d & 1) {
                if (can_place(y + dy[d], x + dx[d])) {
                    return d2c[d];
                }
            }
            return '.';
        }

        // (hy, hx) にいる人間がマス (by, bx) に柵を置いた時、ペットを面積 thresh 以下の領域に収容できるか？
        // 人間が thresh 以下の領域に入るのは許容しない
        bool can_make_prison(int hy, int hx, int by, int bx, int thresh) {
            if (!can_place(by, bx)) return false;
            blocked_tmp[by][bx] = true;

            UnionFind tree(N * N);
            // yoko
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N - 1; x++) {
                    if (blocked_tmp[y][x] || blocked_tmp[y][x + 1]) continue;
                    tree.unite(y * N + x, y * N + x + 1);
                }
            }
            // tate
            for (int y = 0; y < N - 1; y++) {
                for (int x = 0; x < N; x++) {
                    if (blocked_tmp[y][x] || blocked_tmp[y + 1][x]) continue;
                    tree.unite(y * N + x, (y + 1) * N + x);
                }
            }
            int hpos = hy * N + hx;
            if (tree.size(hpos) <= thresh) {
                // 人間が狭い領域に入ってしまう
                blocked_tmp[by][bx] = false;
                return false;
            }

            for (int d = 0; d < 4; d++) {
                int nby = by + dy[d], nbx = bx + dx[d]; // block の 4-neighbor
                if (!is_inside(nby, nbx) || blocked_tmp[nby][nbx]) continue;
                int nbpos = nby * N + nbx;
                int area = tree.size(nbpos);
                if (area > thresh) continue;
                int nbid = tree.find(nbpos);
                // pet
                for (auto [pid, py, px, pt] : pets) {
                    int ppos = py * N + px;
                    if (tree.find(ppos) == nbid) {
                        // dump("trap!", turn, pid);
                        blocked_tmp[by][bx] = false;
                        return true;
                    }
                }
            }

            blocked_tmp[by][bx] = false;
            return false;
        }

        char calc_action(int hid) {
            auto& action_queue = humans[hid].action_queue;
            auto [y, x] = humans[hid].get_coord();

            // 幽閉可能ならする
            for (int d = 0; d < 4; d++) {
                int by = y + dy[d], bx = x + dx[d];
                if (can_make_prison(y, x, by, bx, 40)) {
                    return d2c[d];
                }
            }
            
            if (action_queue.empty()) return '.';
            auto action = action_queue.front();
            auto type = action.type;
            switch (type)
            {
            case Action::Type::MOVE:
                return calc_move(hid, action.y, action.x);
            case Action::Type::BLOCK:
                return calc_block(hid, action.mask);
            case Action::Type::WAIT:
                return '.';
            }
            assert(false);
            return '.';
        }

        string calc_actions() {
            // 各人の行動を決定する過程で制約が増える　blocked_tmp, human_count_tmp でその差分を記録する
            memcpy(blocked_tmp, blocked, sizeof(bool) * N * N);
            memcpy(human_count_tmp, human_count, sizeof(int) * N * N);

            string actions(humans.size(), '.');

            for (int hid = 0; hid < humans.size(); hid++) {

                char action = calc_action(hid);
                auto [y, x] = humans[hid].get_coord();

                if (isupper(action)) { // move
                    int d = c2d[action];
                    human_count_tmp[y + dy[d]][x + dx[d]]++;
                }
                else if (islower(action)) { // block
                    int d = c2d[action];
                    blocked_tmp[y + dy[d]][x + dx[d]] = true;
                }

                actions[hid] = action;
            }

            return actions;
        }

        void do_actions(const string& actions) {
            for (int hid = 0; hid < humans.size(); hid++) {
                int& y = humans[hid].y;
                int& x = humans[hid].x;
                char c = actions[hid];
                if (c == '.') continue;
                if (isupper(c)) {
                    human_count[y][x]--;
                    y += dy[c2d[c]];
                    x += dx[c2d[c]];
                    human_count[y][x]++;
                }
                if (islower(c)) {
                    blocked[y + dy[c2d[c]]][x + dx[c2d[c]]] = true;
                }
            }
        }

        void update_queue() {
            for (int hid = 0; hid < humans.size(); hid++) {
                auto& action_queue = humans[hid].action_queue;
                auto [y, x] = humans[hid].get_coord();
                // 不要な操作を wipe
                while (!action_queue.empty()) {
                    bool updated = false;
                    const auto& action = action_queue.front();
                    auto type = action.type;
                    switch (type)
                    {
                    case Action::Type::MOVE:
                    {
                        if (y == action.y && x == action.x) {
                            action_queue.pop_front();
                            updated = true;
                        }
                        break;
                    }
                    case Action::Type::BLOCK:
                    {
                        bool completed = true;
                        for (int d = 0; d < 4; d++) if (action.mask >> d & 1) {
                            int ny = y + dy[d], nx = x + dx[d];
                            if (!is_inside(ny, nx)) continue;
                            if (!blocked[ny][nx]) {
                                completed = false;
                                break;
                            }
                        }
                        if (completed) {
                            action_queue.pop_front();
                            updated = true;
                        }
                        break;
                    }
                    case Action::Type::WAIT:
                    {
                        if (turn >= action.time) {
                            action_queue.pop_front();
                            updated = true;
                        }
                    }
                    }
                    if (!updated) break;
                }
            }
        }

        bool all_queue_empty() const {
            for (const auto& human : humans) if (!human.action_queue.empty()) return false;
            return true;
        }

        static vector<Task> create_task_list() {

            // TODO: 人間の衝突をどうするか…
            
            auto board = make_vector('.', N, N);
            for (int s = 3; s < 57; s += 3) {
                for (int y = 0; y < N; y++) {
                    for (int x = 0; x < N; x++) {
                        if (y + x == s) {
                            board[y][x] = '#';
                        }
                    }
                }
            }

            {
                int moved = 0, y = 0, x = 0;
                vector<Point> mv({ {2,1},{1,2} });
                while (true) {
                    auto m = mv[moved % 2];
                    y += m.y; x += m.x;
                    if (y + x > 54) break;
                    board[y][x] = 'o';
                    moved++;
                }
            }

            vector<Task> task_list;

            for (int id = 0; id <= 4; id++) {
                Task task;
                task.id = id;
                task.est_cost = 0;

                int offset = id;

                auto& actions = task.actions;
                int& cost = task.est_cost;
                int y = offset * 6 + 5, x = 0;
                actions.push_back(Action::move(y, x)); // 最初の移動はコストに考慮しない
                for (int k = 0; k <= offset * 3; k++) {
                    actions.push_back(Action::block(1 << 3)); cost++;
                    y--;
                    actions.push_back(Action::move(y, x)); cost++;
                    actions.push_back(Action::block(1 << 1)); cost++;
                    x++;
                    actions.push_back(Action::move(y, x)); cost++;
                }
                actions.push_back(Action::block((1 << 0) | (1 << 3))); cost += 2;

                task_list.push_back(task);
            }

            for (int id = 5; id <= 9; id++) {
                Task task;
                task.id = id;
                task.est_cost = 0;

                int offset = id - 5;

                auto& actions = task.actions;
                int& cost = task.est_cost;
                int y = 0, x = offset * 6 + 5;
                actions.push_back(Action::move(y, x)); // 最初の移動はコストに考慮しない
                for (int k = 0; k <= offset * 3 + 1; k++) {
                    actions.push_back(Action::block(1 << 0)); cost++;
                    x--;
                    actions.push_back(Action::move(y, x)); cost++;
                    actions.push_back(Action::block(1 << 2)); cost++;
                    y++;
                    actions.push_back(Action::move(y, x)); cost++;
                }
                actions.push_back(Action::block(1 << 0)); cost++;

                task_list.push_back(task);
            }

            for (int id = 10; id <= 13; id++) {
                Task task;
                task.id = id;
                task.est_cost = 0;

                int offset = id - 10;

                auto& actions = task.actions;
                int& cost = task.est_cost;
                int y = 29, x = offset * 6 + 5;
                actions.push_back(Action::move(y, x)); // 最初の移動はコストに考慮しない
                for (int k = 0; k <= (3 - offset) * 3 + 1; k++) {
                    actions.push_back(Action::block(1 << 2)); cost++;
                    x++;
                    actions.push_back(Action::move(y, x)); cost++;
                    actions.push_back(Action::block(1 << 0)); cost++;
                    y--;
                    actions.push_back(Action::move(y, x)); cost++;
                }
                actions.push_back(Action::block(1 << 2)); cost++;

                task_list.push_back(task);
            }

            for (int id = 14; id <= 17; id++) {
                Task task;
                task.id = id;
                task.est_cost = 0;

                int offset = id - 14;

                auto& actions = task.actions;
                int& cost = task.est_cost;
                int y = offset * 6 + 5, x = 29;
                actions.push_back(Action::move(y, x)); // 最初の移動はコストに考慮しない
                for (int k = 0; k <= (3 - offset) * 3 + 1; k++) {
                    actions.push_back(Action::block(1 << 1)); cost++;
                    y++;
                    actions.push_back(Action::move(y, x)); cost++;
                    actions.push_back(Action::block(1 << 3)); cost++;
                    x--;
                    actions.push_back(Action::move(y, x)); cost++;
                }
                actions.push_back(Action::block((1 << 1) | (1 << 2))); cost += 2;

                task_list.push_back(task);
            }

            return task_list;
        }

        static vector<TrapTask> create_traptask_list() {
            
            int id = 0, npoints = 5;
            vector<TrapTask> tasks;
            for (int sy = 5; sy <= 29; sy += 3) {
                TrapTask task;
                task.id = id++;
                int y = sy, x = 0, np = 0;
                while (true) {
                    task.area.emplace_back(y--, x); np++;
                    if (np == npoints) break;
                    task.area.emplace_back(y, x++); np++;
                    if (np == npoints) break;
                }
                tasks.push_back(task);
                npoints += 3;
            }

            npoints = 27;
            for (int sx = 2; sx <= 23; sx += 3) {
                TrapTask task;
                task.id = id++;
                int y = 29, x = sx, np = 0;
                while (true) {
                    task.area.emplace_back(y, x++); np++;
                    if (np == npoints) break;
                    task.area.emplace_back(y--, x); np++;
                    if (np == npoints) break;
                }
                tasks.push_back(task);
                npoints -= 3;
            }

            npoints = 6;
            for (int sx = 5; sx <= 29; sx += 3) {
                TrapTask task;
                task.id = id++;
                int y = 0, x = sx, np = 0;
                while (true) {
                    task.area.emplace_back(y, x--); np++;
                    if (np == npoints) break;
                    task.area.emplace_back(y++, x); np++;
                    if (np == npoints) break;
                }
                tasks.push_back(task);
                npoints += 3;
            }

            npoints = 28;
            for (int sy = 2; sy <= 23; sy += 3) {
                TrapTask task;
                task.id = id++;
                int y = sy, x = 29, np = 0;
                while (true) {
                    task.area.emplace_back(y++, x); np++;
                    if (np == npoints) break;
                    task.area.emplace_back(y, x--); np++;
                    if (np == npoints) break;
                }
                tasks.push_back(task);
                npoints -= 3;
            }

            return tasks;
        }

        int calc_dist(int sy, int sx, int gy, int gx) const {
            static int dist[N][N];
            Fill(dist, inf);
            std::queue<Point> qu;
            qu.emplace(sy, sx);
            dist[sy][sx] = 0;
            while (!qu.empty()) {
                auto [y, x] = qu.front(); qu.pop();
                for (int d = 0; d < 4; d++) {
                    int ny = y + dy[d], nx = x + dx[d];
                    if (!is_inside(ny, nx) || blocked[ny][nx] || dist[ny][nx] != inf) continue;
                    dist[ny][nx] = dist[y][x] + 1;
                    if (ny == gy && nx == gx) return dist[ny][nx];
                    qu.emplace(ny, nx);
                }
            }
            return inf;
        }

        int get_dir(const Point& p1, const Point& p2) const {
            if (p1.y == p2.y) return p1.x < p2.x ? 0 : 2;
            assert(p1.x == p2.x);
            return p1.y < p2.y ? 3 : 1;
        }

        bool is_valid_traptask(const TrapTask& task) const {
            const auto& area = task.area;
            Point bp[2];
            for (int i = 0; i < 2; i++) {
                auto p1 = area[area.size() - 5 + i], p2 = area[area.size() - 4 + i];
                int d = get_dir(p1, p2);
                bp[i] = Point(p2.y + dy[d], p2.x + dx[d]);
            }
            if (!blocked[bp[0].y][bp[0].x] || !blocked[bp[1].y][bp[1].x]) return false;
            for (int i = 0; i < area.size() - 3; i++) {
                auto p = area[i];
                if (pet_count[p.y][p.x]) {
                    return true;
                }
            }

            return false;
        }

        void solve() {

            vector<bool> base_task_finished(humans.size(), false);

            auto tasks = create_task_list();
            auto traptasks = create_traptask_list();

            auto all_tasks_taken = [&]() {
                for (const auto& task : tasks) if (!task.taken) return false;
                return true;
            };

            auto assign_task = [&]() {
                if (all_tasks_taken()) {
                    // cancel traptask
                    for (auto& task : traptasks) {
                        if (task.taken != -1 && !is_valid_traptask(task)) {
                            dump("cancel", turn, task.taken);
                            humans[task.taken].action_queue.clear();
                            task.taken = -1;
                        }
                    }
                    // take traptask
                    for (auto& task : traptasks) {
                        if (task.taken == -1 && is_valid_traptask(task)) {
                            // 最も近いフリーの人間を向かわせる
                            const auto& area = task.area;
                            int min_dist = inf, hid = -1;
                            for (const auto& human : humans) {
                                auto [sy, sx] = human.get_coord();
                                auto [gy, gx] = area.back();
                                int dist = calc_dist(sy, sx, gy, gx);
                                if (chmin(min_dist, dist)) {
                                    hid = human.id;
                                }
                            }
                            if (hid != -1) {
                                auto& human = humans[hid];
                                task.taken = hid;
                                auto hp = area.back();
                                auto bp = area[area.size() - 2];
                                int d = get_dir(hp, bp);
                                human.action_queue.push_back(Action::move(hp));
                                human.action_queue.push_back(Action::block(1 << d));   
                            }
                        }
                    }
                    return;
                }
                for (auto& human : humans) if (human.action_queue.empty()) {
                    int min_dist = inf, selected_task_id = -1;
                    auto [sy, sx] = human.get_coord();
                    for (const auto& task : tasks) if (!task.taken) {
                        int gy = task.actions.front().y, gx = task.actions.front().x;
                        int dist = calc_dist(sy, sx, gy, gx);
                        if (chmin(min_dist, dist)) {
                            selected_task_id = task.id;
                        }
                    }
                    if (selected_task_id != -1) {
                        auto& task = tasks[selected_task_id];
                        std::copy(task.actions.begin(), task.actions.end(), std::back_inserter(human.action_queue));
                        task.taken = true;
                        //dump(human.id, task.id, min_dist + task.est_cost);
                    }
                }
            };

            assign_task();
            update_queue();
            while (turn < MAX_TURN) {
                auto actions = calc_actions();
                do_actions(actions);
                cout << actions << endl;
                load();
                update_queue();
                assign_task();
                turn++;
            }

        }

    };

}

namespace NSolver2 {
    // 座標は 30x30 に境界を追加した 32x32 を 1dim に直した [0,1024) で表す

    using ::operator<<;

    constexpr int N = 32;
    constexpr int NN = N * N;
    constexpr int dir[] = { 1, -N, -1, N };

    struct coord {
        int idx;
        coord(int idx = 0) : idx(idx) {}
        coord(int y, int x) : idx(x | y << 5) {}
        inline int y() const { return idx >> 5; }
        inline int x() const { return idx & 0b11111; }
        inline std::pair<int, int> unpack() const { return { y(), x() }; }
        inline int operator[] (int i) const { return i ? x() : y(); }
        inline coord& move(int d) { idx += dir[d]; return *this; }
        inline coord& move(char c) { return move(c2d[tolower(c)]); }
        inline coord& move(const string& s) { for (char c : s) move(c); return *this; }
        inline coord moved(int d) const { return coord(*this).move(d); }
        inline coord moved(char c) const { return moved(c2d[tolower(c)]); }
        bool operator==(const coord& c) { return idx == c.idx; }
        string stringify() const { return format("(%d,%d)", y(), x()); }
        friend std::istream& operator>>(std::istream& in, coord& pos) {
            int y, x;
            in >> y >> x;
            pos = coord(y, x);
            return in;
        }
    };

    struct Action {

        enum struct Type { MOVE, BLOCK, WAIT };

        static Action move(coord crd, int task_id = -1) { return Action(Type::MOVE, crd.idx, task_id); }
        static Action block(int dir, int task_id = -1) { return Action(Type::BLOCK, dir, task_id); }
        static Action wait(int until) { return Action(Type::WAIT, until, -1); }

        inline Type get_type() const { return type; }
        inline coord get_coord() const { return coord(data); }
        inline int get_dir() const { return data; }
        inline int get_time() const { return data; }
        inline int get_tid() const { return task_id; }

        string stringify() const {
            string taskstr;
            if (task_id != -1) taskstr += format("Task%d: ", task_id);
            switch (type) {
            case Type::MOVE:  return taskstr + "move to " + coord(data).stringify();
            case Type::BLOCK: return taskstr + "block " + d2C[data];
            case Type::WAIT:  return taskstr + "wait until " + std::to_string(data);
            }
            return "";
        }

    private:

        Type type;
        int data;
        int task_id;

        Action(Type type, int data, int task_id) : type(type), data(data), task_id(task_id) {}

    };

    struct Human;

    struct Task {
        enum struct Type {
            SEQUENTIAL, // action 列を逐次的に実行するタスク
            CAPTURE     // ペット捕獲タスク
        };
        int id;
        Type type;
        Human* assignee;
        bool is_cancelable; // sequential は不可
        bool is_completed;
        virtual bool proceed() { return false; }
    };

    struct SequentialTask : Task {
        int progress;
        vector<Action> actions;
        bool proceed() override { return ++progress == actions.size(); }
    };

    struct TaskGenerator {

        int ctr_id = 0;

        // 初期位置とコマンド列から action list を生成
        SequentialTask generate_sequential_task(coord pos, const string& cmd_list) {

            SequentialTask task;
            task.id = ctr_id++;
            task.type = Task::Type::SEQUENTIAL;
            task.assignee = nullptr;
            task.is_cancelable = false;
            task.is_completed = false;
            task.progress = 0;

            auto& actions = task.actions;
            actions.push_back(Action::move(pos, task.id));
            for (char c : cmd_list) {
                assert(c != '.'); // wait は許容しない
                if (isupper(c)) { // move
                    pos.move(c);
                    actions.push_back(Action::move(pos, task.id));
                }
                else { // block
                    actions.push_back(Action::block(c2d[c], task.id));
                }
            }

            return task;
        }

        vector<SequentialTask> generate_sequential_tasks() {

            vector<SequentialTask> tasks;

            auto rep = [](int n, const string& s) {
                string res;
                while (n--) res += s;
                return res;
            };

            for (int k = 0; k < 5; k++) {
                tasks.push_back(generate_sequential_task({ 6 * k + 6 , 1 }, rep(3 * k + 1, "dUuR") + "dr"));
            }
            for (int k = 0; k < 5; k++) {
                tasks.push_back(generate_sequential_task({ 1, 6 * k + 6 }, rep(3 * k + 2, "rLlD") + "r"));
            }
            for (int k = 0; k < 4; k++) {
                tasks.push_back(generate_sequential_task({ 30, 24 - 6 * k }, rep(3 * k + 2, "lRrU") + "l"));
            }
            for (int k = 0; k < 4; k++) {
                tasks.push_back(generate_sequential_task({ 24 - 6 * k, 30 }, rep(3 * k + 2, "uDdL") + "ul"));
            }

            return tasks;
        }

    };

    struct Pet {
        enum struct Type { COW, PIG, RABBIT, DOG, CAT };
        int id;
        coord pos;
        Type type;
        Pet(int id = -1, coord pos = -1, Type type = Type(-1)) : id(id), pos(pos), type(type) {}
        Pet(int id = -1, coord pos = -1, int type = -1) : id(id), pos(pos), type(Type(type)) {}
        string stringify() const { return format("Pet[%d,%s,%s]", id, pos.stringify().c_str(), type_str().c_str()); }
    private:
        string type_str() const {
            switch (type) {
            case Type::COW:     return "Cow";
            case Type::PIG:     return "Pig";
            case Type::RABBIT:  return "Rabbit";
            case Type::DOG:     return "Dog";
            case Type::CAT:     return "Cat";
            }
            return "undefined";
        }
    };

    struct Human {
        int id;
        coord pos;
        Task* task;
        std::deque<Action> qu;
        Human(int id = -1, coord pos = -1) : id(id), pos(pos), task(nullptr) {}
        void assign(SequentialTask* task_) {
            this->task = task_;
            task->assignee = this;
            std::copy(task_->actions.begin(), task_->actions.end(), std::back_inserter(qu));
        }
        string stringify() const { return format("Human[%d,%s]", id, pos.stringify().c_str()); }
    };

    struct State {

        std::istream& in;
        std::ostream& out;

        int turn;

        vector<Pet> pets;
        vector<Human> humans;

        bool is_blocked[NN];
        int ctr_human[NN];
        int ctr_pet[NN];

        // 行動途中の情報の記録
        bool is_blocked_tmp[NN];
        int ctr_human_tmp[NN];

        State(std::istream& in, std::ostream& out) : in(in), out(out) { init(); }

        void init() {
            Fill(is_blocked, false);
            for (int y = 0; y < N; y++) is_blocked[y * N] = is_blocked[y * N + N - 1] = true;
            for (int x = 0; x < N; x++) is_blocked[x] = is_blocked[N * (N - 1) + x] = true; // 境界
            Fill(ctr_human, 0);
            Fill(ctr_pet, 0);
            turn = 0;
            int num_pets; in >> num_pets;
            for (int pid = 0; pid < num_pets; pid++) {
                coord pos;
                int t;
                cin >> pos >> t;
                pets.emplace_back(pid, pos, t);
                ctr_pet[pos.idx]++;
            }
            int num_humans; in >> num_humans;
            for (int hid = 0; hid < num_humans; hid++) {
                coord pos;
                cin >> pos;
                humans.emplace_back(hid, pos);
                ctr_human[pos.idx]++;
            }
        }

        void move(Pet& pet, char c) {
            ctr_pet[pet.pos.idx]--;
            pet.pos.move(c);
            ctr_pet[pet.pos.idx]++;
        }

        void move(Pet& pet, const string& s) {
            for (char c : s) move(pet, c);
        }

        void load() {
            vector<string> moves(pets.size());
            cin >> moves;
            for (int pid = 0; pid < pets.size(); pid++) {
                move(pets[pid], moves[pid]);
            }
        }

        bool can_block(coord pos) const {
            // 重複して設置しない || 人間に刺さない || ペットに刺さない
            int idx = pos.idx;
            if (is_blocked_tmp[idx] || ctr_human_tmp[idx] || ctr_pet[idx]) return false;
            for (int d = 0; d < 4; d++) {
                int nidx = idx + dir[d];
                if (ctr_pet[nidx]) return false; // ペットの 4 近傍に置かない
            }
            return true;
        }

        // 人間が target に最短経路で移動するための移動方向（候補複数ならランダム）
        char calc_move(const Human& human, const Action& action) {
            static int dist[NN];

            assert(action.get_type() == Action::Type::MOVE);

            int sidx = human.pos.idx, gidx = action.get_coord().idx;
            assert(sidx != gidx);
            
            // pet, human は無視して目的地からの距離を計算
            Fill(dist, inf);
            std::queue<int> qu;
            qu.emplace(gidx);
            dist[gidx] = 0;
            while (!qu.empty()) {
                int idx = qu.front(); qu.pop();
                for (int d = 0; d < 4; d++) {
                    int nidx = idx + dir[d];
                    // 設置予定の柵も含む
                    if (is_blocked_tmp[nidx] || dist[nidx] != inf) continue;
                    dist[nidx] = dist[idx] + 1;
                    qu.emplace(nidx);
                }
            }

            if (dist[sidx] == inf) return '.'; // 移動不可

            // 最短経路移動方向の候補を調べる
            int min_dist = inf;
            vector<int> cands;
            for (int d = 0; d < 4; d++) {
                int nidx = sidx + dir[d];
                // 柵とペットを踏まない
                // 人間については、元からいた場所は避ける (移動先が重複するのは許す)
                if (is_blocked_tmp[nidx] || ctr_human[nidx] || ctr_pet[nidx] || dist[nidx] > min_dist) continue;
                if (dist[nidx] < min_dist) {
                    min_dist = dist[nidx];
                    cands.clear();
                }
                cands.push_back(d);
            }

            if (min_dist == inf) return '.'; // 移動不可 (pet に包囲されるケースが稀にある)

            // ランダムで選ぶ
            int d = cands[rnd.next_int(cands.size())];
            return d2C[d];
        }

        // 柵の設置
        char calc_block(const Human& human, const Action& action) {
            assert(action.get_type() == Action::Type::BLOCK);
            int d = action.get_dir();
            auto pos = human.pos.moved(d);
            return can_block(pos) ? d2c[d] : '.';
        }
        
        // (hy, hx) にいる人間がマス (by, bx) に柵を置いた時、ペットを面積 thresh 以下の領域に収容できるか？
        // 人間が trap 側領域に入るのは許容しない
        bool can_trap(const Human& human, int d, int thresh) {
            auto bpos = human.pos.moved(d);
            if (!can_block(bpos)) return false;
            is_blocked_tmp[bpos.idx] = true;

            UnionFind tree(NN);
            // yoko
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N - 1; x++) {
                    if (is_blocked_tmp[y * N + x] || is_blocked_tmp[y * N + x + 1]) continue;
                    tree.unite(y * N + x, y * N + x + 1);
                }
            }
            // tate
            for (int y = 0; y < N - 1; y++) {
                for (int x = 0; x < N; x++) {
                    if (is_blocked_tmp[y * N + x] || is_blocked_tmp[(y + 1) * N + x]) continue;
                    tree.unite(y * N + x, (y + 1) * N + x);
                }
            }

            for (const auto& h : humans) {
                if (tree.size(h.pos.idx) <= thresh) {
                    // 人間が狭い領域に入ってしまう
                    is_blocked_tmp[bpos.idx] = false;
                    return false;
                }
            }

            for (int d = 0; d < 4; d++) {
                coord npos = bpos.moved(d);
                if (is_blocked_tmp[npos.idx]) continue;
                int area = tree.size(npos.idx);
                if (area > thresh) continue;
                int nbid = tree.find(npos.idx);
                // pet
                for (auto [pid, ppos, pt] : pets) {
                    if (tree.find(ppos.idx) == nbid) {
                        dump("trap!", turn, pid);
                        is_blocked_tmp[bpos.idx] = false;
                        return true;
                    }
                }
            }

            is_blocked_tmp[bpos.idx] = false;
            return false;
        }

        char calc_action(const Human& human) {
            auto& qu = human.qu;

            // 幽閉可能ならする
            for (int d = 0; d < 4; d++) {
                if (can_trap(human, d, 40)) {
                    return d2c[d];
                }
            }

            if (!human.task) {
                auto pos = human.pos;
                int d = rnd.next_int(4);
                auto action = Action::move(pos.moved(d));
                return calc_move(human, action);
            }

            auto action = qu.front();
            auto type = action.get_type();
            switch (type) {
            case Action::Type::MOVE:
                return calc_move(human, action);
            case Action::Type::BLOCK:
                return calc_block(human, action);
            case Action::Type::WAIT:
                return '.';
            }
            assert(false);
            return '.';
        }

        string calc_actions() {
            // 各人の行動を決定する過程で制約が増える　blocked_tmp, human_count_tmp でその差分を記録する
            memcpy(is_blocked_tmp, is_blocked, sizeof(bool) * NN);
            memcpy(ctr_human_tmp, ctr_human, sizeof(int) * NN);

            string actions(humans.size(), '.');

            for (int hid = 0; hid < humans.size(); hid++) {

                char action = calc_action(humans[hid]);
                const auto& pos = humans[hid].pos;
                if (isupper(action)) { // move
                    int d = c2d[action];
                    ctr_human_tmp[pos.moved(d).idx]++;
                }
                else if (islower(action)) { // block
                    int d = c2d[action];
                    is_blocked_tmp[pos.moved(d).idx] = true;
                }

                actions[hid] = action;
            }

            return actions;
        }

        void do_actions(const string& actions) {

            for (int hid = 0; hid < humans.size(); hid++) {
                auto& pos = humans[hid].pos;
                char c = actions[hid];
                if (c == '.') continue;
                if (isupper(c)) {
                    ctr_human[pos.idx]--;
                    pos.move(c);
                    ctr_human[pos.idx]++;
                }
                if (islower(c)) {
                    is_blocked[pos.moved(c).idx] = true;
                }
            }
        }

        void update_queue(Human& human) {
            auto& [id, pos, task, qu] = human;
            while (!qu.empty()) {
                bool updated = false;
                const auto& act = qu.front();
                auto type = act.get_type();
                switch (type) {
                case Action::Type::MOVE:
                {
                    if (pos == act.get_coord()) {
                        if (task && act.get_tid() == task->id) {
                            if (task->proceed()) {
                                task->assignee = nullptr;
                                task->is_completed = true;
                                task = nullptr;
                            }
                        }
                        qu.pop_front();
                        updated = true;
                    }
                    break;
                }
                case Action::Type::BLOCK:
                {
                    if (is_blocked[pos.moved(act.get_dir()).idx]) {
                        if (task && act.get_tid() == task->id) {
                            if (task->proceed()) {
                                task->assignee = nullptr;
                                task->is_completed = true;
                                task = nullptr;
                            }
                        }
                        qu.pop_front();
                        updated = true;
                    }
                    break;
                }
                case Action::Type::WAIT:
                {
                    if (turn >= act.get_time()) {
                        qu.pop_front();
                        updated = true;
                    }
                    break;
                }
                }
                if (!updated) break;
            }
        }

        void update_queue() {
            for (auto& human : humans) {
                update_queue(human);
            }
        }

        int calc_dist(coord from, coord to) const {
            static int dist[NN];
            Fill(dist, inf);
            std::queue<coord> qu;
            qu.emplace(from);
            dist[from.idx] = 0;
            while (!qu.empty()) {
                auto u = qu.front(); qu.pop();
                for (int d = 0; d < 4; d++) {
                    auto v = u.moved(d);
                    if (is_blocked[v.idx] || dist[v.idx] != inf) continue;
                    dist[v.idx] = dist[u.idx] + 1;
                    if (v == to) return dist[v.idx];
                    qu.emplace(v);
                }
            }
            return inf;
        }

        void solve() {

            // 柵の設置
            // ターン開始時点でペットもしくは人が居るマスを選ぶことは出来ない
            // 加えて隣接するマスにペットが存在する場合も選ぶことは出来ない
            
            // 人間の移動
            // 柵(そのターンに設置されるものも含む)以外ならどこでも通行可能

            auto tgen = TaskGenerator();
            auto tasks = tgen.generate_sequential_tasks();

            auto assign_task = [&]() {
                for (auto& human : humans) if (!human.task) {
                    int min_dist = inf;
                    SequentialTask* selected_task = nullptr;
                    auto from = human.pos;
                    for (auto& task : tasks) if (!task.assignee && !task.is_completed) {
                        auto to = task.actions.front().get_coord();
                        int dist = calc_dist(from, to);
                        if (chmin(min_dist, dist)) selected_task = &task;
                    }
                    if (selected_task) {
                        human.assign(selected_task);
                        //dump(turn, human.id, selected_task->id, selected_task->actions);
                    }
                }
            };

            assign_task();
            update_queue();
            while (turn < MAX_TURN) {
                auto actions = calc_actions();
                do_actions(actions);
                cout << actions << endl;
                load();
                update_queue();
                assign_task();
                turn++;
            }

        }

    };

    void sandbox() {

        TaskGenerator gen;
        auto tasks = gen.generate_sequential_tasks();
        
    }

}

#ifdef HAVE_OPENCV_HIGHGUI

namespace NManual {

    constexpr int N = 30;
    constexpr int dy[] = { 0, -1, 0, 1 };
    constexpr int dx[] = { 1, 0, -1, 0 };

    cv::Mat_<cv::Vec3b> get_empty_icon(const cv::Size& size, const cv::Vec3b bgcolor = cv::Vec3b(200, 200, 200)) {
        cv::Mat_<cv::Vec3b> img(size.height, size.height, bgcolor);
        cv::rectangle(img, cv::Rect(0, 0, size.width, size.height), cv::Scalar(255, 255, 255));
        return img;
    }

    cv::Mat_<cv::Vec3b> load_icon(
        const string& path, const cv::Size& size, const cv::Vec3b bgcolor = cv::Vec3b(200, 200, 200)
    ) {
        cv::Size in_size(size.width - 2, size.height - 2);
        auto img = cv::imread(path, -1);
        cv::Mat_<cv::Vec3b> img2(img.rows, img.cols, bgcolor);
        for (int y = 0; y < img2.rows; y++) {
            for (int x = 0; x < img2.cols; x++) {
                int alpha = img.at<cv::Vec4b>(y, x)[3];
                if (alpha < 255) continue;
                auto px = img.at<cv::Vec4b>(y, x);
                img2.at<cv::Vec3b>(y, x) = cv::Vec3b(px[0], px[1], px[2]);
            }
        }
        cv::Mat_<cv::Vec3b> img3;
        cv::resize(img2, img3, in_size, 0, 0, cv::INTER_NEAREST);
        cv::Mat_<cv::Vec3b> img4(size.height, size.width, cv::Vec3b(255, 255, 255));
        img3.copyTo(img4(cv::Rect(1, 1, in_size.width, in_size.height)));
        return img4;
    }

    constexpr int icon_size = 32;
    constexpr int button_size = 64;
    const auto img_size = cv::Size(icon_size, icon_size);
    const auto icon_empty = get_empty_icon(img_size);
    const auto icon_human = load_icon("img/human.png", img_size);
    const auto icon_block = load_icon("img/block.png", img_size);
    const auto icon_cow = load_icon("img/cow.png", img_size);
    const auto icon_pig = load_icon("img/pig.png", img_size);
    const auto icon_rabbit = load_icon("img/rabbit.png", img_size);
    const auto icon_dog = load_icon("img/dog.png", img_size);
    const auto icon_cat = load_icon("img/cat.png", img_size);
    const cv::Mat_<cv::Vec3b> icon_pets[5] = { icon_cow, icon_pig, icon_rabbit, icon_dog, icon_cat };
    const cv::Mat_<cv::Vec3b> icon_all[8] = { icon_cow, icon_pig, icon_rabbit, icon_dog, icon_cat, icon_human, icon_block, icon_empty };

    struct State {

        struct Pet {
            int id;
            int y, x;
            int type;
            int target_id; // for dog
            int target_y, target_x; // for cat
            Pet(int id = -1, int y = -1, int x = -1, int t = -1, int tid = -1, int ty = -1, int tx = -1)
                : id(id), y(y), x(x), type(t), target_id(tid), target_y(ty), target_x(tx) {}
        };

        struct Human {
            int id, y, x;
            Human(int id = -1, int y = -1, int x = -1) : id(id), y(y), x(x) {}
        };

        Xorshift rnd;

        int pet_counter;
        vector<Pet> pets;

        int human_counter;
        vector<Human> humans;

        bool is_blocked[N][N];

        int selected = -1;

        State(std::istream& in, std::ostream& out) { 
            load(in, out);
        }

        inline bool is_inside(int y, int x) const {
            return 0 <= y && y < N && 0 <= x && x < N;
        }

        cv::Mat_<cv::Vec3b> create_palette_img() const {
            int H = button_size;
            int W = button_size * 7; // pet*5, human, block
            cv::Mat_<cv::Vec3b> img(H, W, cv::Vec3b(255, 255, 255));
            for (int c = 0; c < 7; c++) {
                cv::Rect roi(c * button_size, 0, button_size, button_size);
                cv::Mat_<cv::Vec3b> img_roi;
                cv::resize(icon_all[c], img_roi, cv::Size(button_size, button_size), 0, 0, cv::INTER_NEAREST);
                if (c == selected) {
                    cv::rectangle(img_roi, cv::Rect(0, 0, button_size, button_size), cv::Scalar(0, 0, 255));
                }
                img_roi.copyTo(img(roi));
            }
            return img;
        }

        static void palette_callback(int e, int x, int y, int f, void* param) {
            State* state = static_cast<State*>(param);
            if (e == 4) {
                int c = x / button_size;
                state->selected = c;
                cerr << c << endl;
            }
        }

        static void board_callback(int e, int x, int y, int f, void* param) {
            if (e != 4 && e != 5) return;
            State* state = static_cast<State*>(param);
            if (state->selected == -1) return;

            int selected = state->selected;
            int cy = y / icon_size, cx = x / icon_size;
            dump(cy, cx);

            // delete
            for (auto it = state->humans.begin(); it != state->humans.end(); ++it) {
                if (it->y == cy && it->x == cx) {
                    state->humans.erase(it);
                    break;
                }
            }
            for (auto it = state->pets.begin(); it != state->pets.end(); ++it) {
                if (it->y == cy && it->x == cx) {
                    state->pets.erase(it);
                    break;
                }
            }
            if (state->is_blocked[cy][cx]) state->is_blocked[cy][cx] = false;

            if (e == 5) return;

            // append
            if (selected < 5) state->pets.emplace_back(state->pet_counter++, cy, cx, selected);
            else if (selected == 5) state->humans.emplace_back(state->human_counter++, cy, cx);
            else if (selected == 6) state->is_blocked[cy][cx] = true;
        }

        void load(std::istream& in, std::ostream& out) {
            Fill(is_blocked, false);
            int num_pets; in >> num_pets;
            pet_counter = 0;
            for (int pid = 0; pid < num_pets; pid++) {
                int y, x, t;
                cin >> y >> x >> t;
                y--; x--; t--;
                pets.emplace_back(pet_counter++, y, x, t);
            }
            int num_humans; in >> num_humans;
            human_counter = 0;
            for (int hid = 0; hid < num_humans; hid++) {
                int y, x;
                cin >> y >> x;
                y--; x--;
                humans.emplace_back(human_counter++, y, x);
            }
        }

        cv::Mat_<cv::Vec3b> create_board_img() const {
            cv::Mat_<cv::Vec3b> img(icon_size * N, icon_size * N, cv::Vec3b(0, 0, 0));
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    auto roi = cv::Rect(x * icon_size, y * icon_size, icon_size, icon_size);
                    if (is_blocked[y][x]) {
                        icon_block.copyTo(img(roi));
                    }
                    else {
                        icon_empty.copyTo(img(roi));
                    }
                }
            }
            for (auto [id, y, x, t, tid, ty, tx] : pets) {
                auto roi = cv::Rect(x * icon_size, y * icon_size, icon_size, icon_size);
                icon_pets[t].copyTo(img(roi));
            }
            for (auto [id, y, x] : humans) {
                auto roi = cv::Rect(x * icon_size, y * icon_size, icon_size, icon_size);
                icon_human.copyTo(img(roi));
            }
            return img;
        }

        void do_default_move(Pet& pet) {
            int& y = pet.y;
            int& x = pet.x;
            vector<int> cands;
            for (int d = 0; d < 4; d++) {
                int ny = y + dy[d], nx = x + dx[d];
                if (!is_inside(ny, nx) || is_blocked[ny][nx]) continue;
                cands.push_back(d);
            }
            int d = cands[rnd.next_int(cands.size())];
            y += dy[d];
            x += dx[d];
        }

        void do_dog_move(Pet& pet) {
            static constexpr int inf = INT_MAX / 8;
            static int dist[N][N];

            assert(pet.type == 3);
            const Human* target = nullptr;
            for (const auto& human : humans) {
                if (human.id == pet.target_id) {
                    target = &human;
                }
            }
            if (target && pet.y == target->y && pet.x == target->x) {
                pet.target_id = -1;
                target = nullptr; // 到達済み
            }
            {
                // ペット側から到達可能な位置をチェック
                using pii = std::pair<int, int>;
                // 到達可能か？
                Fill(dist, inf);
                std::queue<pii> qu;
                qu.emplace(pet.y, pet.x);
                dist[pet.y][pet.x] = 0;
                while (!qu.empty()) {
                    auto [y, x] = qu.front(); qu.pop();
                    for (int d = 0; d < 4; d++) {
                        int ny = y + dy[d], nx = x + dx[d];
                        if (!is_inside(ny, nx) || is_blocked[ny][nx] || dist[ny][nx] != inf) continue;
                        qu.emplace(ny, nx);
                        dist[ny][nx] = dist[y][x] + 1;
                    }
                }
            }
            // 現在の target は到達可能か？
            if (target && dist[target->y][target->x] == inf) {
                pet.target_id = -1;
                target = nullptr;
            }
            if (!target) {
                // target 選択
                dist[pet.y][pet.x] = inf; // 現在位置は候補から除外
                vector<const Human*> cands;
                for (const auto& human : humans) {
                    if (dist[human.y][human.x] != inf) {
                        cands.push_back(&human);
                    }
                }
                if (cands.empty()) {
                    // 候補なし: 基本行動
                    pet.target_id = -1;
                    do_default_move(pet);
                    return;
                }
                target = cands[rnd.next_int(cands.size())];
                pet.target_id = target->id;
            }
            assert(target);
            dump(target->id, target->y, target->x, pet.y, pet.x);
            {
                // target 側から bfs
                using pii = std::pair<int, int>;
                Fill(dist, inf);
                std::queue<pii> qu;
                qu.emplace(target->y, target->x);
                dist[target->y][target->x] = 0;
                while (!qu.empty()) {
                    auto [y, x] = qu.front(); qu.pop();
                    for (int d = 0; d < 4; d++) {
                        int ny = y + dy[d], nx = x + dx[d];
                        if (!is_inside(ny, nx) || is_blocked[ny][nx] || dist[ny][nx] != inf) continue;
                        qu.emplace(ny, nx);
                        dist[ny][nx] = dist[y][x] + 1;
                    }
                }
                int min_dist = inf;
                vector<int> cands;
                for (int d = 0; d < 4; d++) {
                    int ny = pet.y + dy[d], nx = pet.x + dx[d];
                    if (!is_inside(ny, nx) || is_blocked[ny][nx] || dist[ny][nx] > min_dist) continue;
                    if (dist[ny][nx] < min_dist) {
                        min_dist = dist[ny][nx];
                        cands.clear();
                    }
                    cands.push_back(d);
                }
                int d = cands[rnd.next_int(cands.size())];
                pet.y += dy[d];
                pet.x += dx[d];
                if (pet.y == target->y && pet.x == target->x) pet.target_id = -1;
                do_default_move(pet);
                if (pet.y == target->y && pet.x == target->x) pet.target_id = -1;
            }
        }

        void do_cat_move(Pet& pet) {
            static constexpr int inf = INT_MAX / 8;
            static int dist[N][N];
            using pii = std::pair<int, int>;

            assert(pet.type == 4);

            {
                // 現在ペット位置から到達可能な点をチェック
                Fill(dist, inf);
                std::queue<pii> qu;
                dist[pet.y][pet.x] = 0;
                qu.emplace(pet.y, pet.x);
                while (!qu.empty()) {
                    auto [y, x] = qu.front(); qu.pop();
                    for (int d = 0; d < 4; d++) {
                        int ny = y + dy[d], nx = x + dx[d];
                        if (!is_inside(ny, nx) || is_blocked[ny][nx] || dist[ny][nx] != inf) continue;
                        qu.emplace(ny, nx);
                        dist[ny][nx] = dist[y][x] + 1;
                    }
                }
            }

            if (
                pet.target_y == -1 ||
                (pet.target_y == pet.y && pet.target_x == pet.x) ||
                dist[pet.target_y][pet.target_x] == inf
                ) 
            {
                pet.target_y = pet.target_x = -1;
            }

            if (pet.target_y == -1) {
                dist[pet.y][pet.x] = inf;
                vector<pii> cands;
                for (int y = 0; y < N; y++) {
                    for (int x = 0; x < N; x++) {
                        if (dist[y][x] != inf) {
                            cands.emplace_back(y, x);
                        }
                    }
                }
                std::tie(pet.target_y, pet.target_x) = cands[rnd.next_int(cands.size())];
            }

            {
                // target 側から bfs
                using pii = std::pair<int, int>;
                Fill(dist, inf);
                std::queue<pii> qu;
                qu.emplace(pet.target_y, pet.target_x);
                dist[pet.target_y][pet.target_x] = 0;
                while (!qu.empty()) {
                    auto [y, x] = qu.front(); qu.pop();
                    for (int d = 0; d < 4; d++) {
                        int ny = y + dy[d], nx = x + dx[d];
                        if (!is_inside(ny, nx) || is_blocked[ny][nx] || dist[ny][nx] != inf) continue;
                        qu.emplace(ny, nx);
                        dist[ny][nx] = dist[y][x] + 1;
                    }
                }
                int min_dist = inf;
                vector<int> cands;
                for (int d = 0; d < 4; d++) {
                    int ny = pet.y + dy[d], nx = pet.x + dx[d];
                    if (!is_inside(ny, nx) || is_blocked[ny][nx] || dist[ny][nx] > min_dist) continue;
                    if (dist[ny][nx] < min_dist) {
                        min_dist = dist[ny][nx];
                        cands.clear();
                    }
                    cands.push_back(d);
                }
                int d = cands[rnd.next_int(cands.size())];
                pet.y += dy[d];
                pet.x += dx[d];
                if (pet.y == pet.target_y && pet.x == pet.target_x) pet.target_y = pet.target_x = -1;
                do_default_move(pet);
                if (pet.y == pet.target_y && pet.x == pet.target_x) pet.target_y = pet.target_x = -1;
            }
        }

        void tick() {
            for (auto& pet : pets) {
                if (pet.type < 3) { // cow, pig, rabbit
                    for (int i = 0; i <= pet.type; i++) {
                        do_default_move(pet);
                    }
                }
                else if (pet.type == 3) { // dog
                    do_dog_move(pet);
                }
                else if (pet.type == 4) { // cat
                    do_cat_move(pet);
                }
            }
        }

        void play() {
            cv::namedWindow("palette", cv::WINDOW_AUTOSIZE);
            cv::namedWindow("board", cv::WINDOW_AUTOSIZE);
            cv::setMouseCallback("palette", palette_callback, this);
            cv::setMouseCallback("board", board_callback, this);

            cv::imshow("palette", create_palette_img());
            cv::imshow("board", create_board_img());

            while (true) {
                int c = cv::waitKeyEx(15);
                if (c == 27) break;
                if (c == 't') tick();

                cv::imshow("palette", create_palette_img());
                cv::imshow("board", create_board_img());
            }
        }

    };

}

#endif

int main() {

    //while (timer.elapsed_ms() < 20000);

    c2d['R'] = c2d['r'] = 0;
    c2d['U'] = c2d['u'] = 1;
    c2d['L'] = c2d['l'] = 2;
    c2d['D'] = c2d['d'] = 3;

    //NManual::State state(cin, cout);
    //state.play();

    //NSolver::State state(cin, cout);
    //state.solve();

    NSolver2::State state(cin, cout);
    state.solve();

    return 0;
}