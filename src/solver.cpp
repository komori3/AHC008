#define _CRT_NONSTDC_NO_WARNINGS
#include <bits/stdc++.h>
#include <random>
#ifdef _MSC_VER
#define ENABLE_VIS
#define ENABLE_DUMP
//#define ENABLE_STATS_DUMP
#endif
#ifdef _MSC_VER
#include <conio.h>
#include <ppl.h>
#ifdef ENABLE_VIS
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif
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

// next_combination : http://yak-ex.blogspot.jp/2014/05/c-nextcombination.html
namespace {

    // possible implementation introduced at http://en.cppreference.com/w/cpp/algorithm/rotate with slight modification to handle parted ranges
    template<typename FI>
    void parted_rotate(FI first1, FI last1, FI first2, FI last2)
    {
        if (first1 == last1 || first2 == last2) return;
        FI next = first2;
        while (first1 != next) {
            std::iter_swap(first1++, next++);
            if (first1 == last1) first1 = first2;
            if (next == last2) {
                next = first2;
            }
            else if (first1 == first2) {
                first2 = next;
            }
        }
    }

    template<typename BI>
    bool next_combination_imp(BI first1, BI last1, BI first2, BI last2)
    {
        if (first1 == last1 || first2 == last2) return false;
        auto target = last1; --target;
        auto last_elem = last2; --last_elem;
        // find right-most incrementable element: target
        while (target != first1 && !(*target < *last_elem)) --target;
        if (target == first1 && !(*target < *last_elem)) {
            parted_rotate(first1, last1, first2, last2);
            return false;
        }
        // find the next value to be incremented: *next
        auto next = first2;
        while (!(*target < *next)) ++next;
        std::iter_swap(target++, next++);
        parted_rotate(target, last1, next, last2);
        return true;
    }

    // INVARIANT: is_sorted(first, mid) && is_sorted(mid, last)
    template<typename BI>
    inline bool next_combination(BI first, BI mid, BI last)
    {
        return next_combination_imp(first, mid, mid, last);
    }

    // INVARIANT: is_sorted(first, mid) && is_sorted(mid, last)
    template<typename BI>
    inline bool prev_combination(BI first, BI mid, BI last)
    {
        return next_combination_imp(mid, last, first, mid);
    }

}


constexpr int MAX_TURN = 300;
constexpr char d2C[5] = "RULD";
constexpr char d2c[5] = "ruld";
int c2d[256];
constexpr int inf = INT_MAX / 8;
// 8-neighbor での連結成分数
// 2 未満なら柵を置いても連結成分数は変化しない
constexpr int cc_table[256] = {
    0,1,1,1,1,1,1,1,1,2,2,2,1,1,1,1,
    1,2,2,2,1,1,1,1,1,2,2,2,1,1,1,1,
    1,2,2,2,2,2,2,2,2,3,3,3,2,2,2,2,
    1,2,2,2,1,1,1,1,1,2,2,2,1,1,1,1,
    1,1,2,1,2,1,2,1,2,2,3,2,2,1,2,1,
    1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,
    1,1,2,1,2,1,2,1,2,2,3,2,2,1,2,1,
    1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,
    1,1,2,1,2,1,2,1,2,2,3,2,2,1,2,1,
    2,2,3,2,2,1,2,1,2,2,3,2,2,1,2,1,
    2,2,3,2,3,2,3,2,3,3,4,3,3,2,3,2,
    2,2,3,2,2,1,2,1,2,2,3,2,2,1,2,1,
    1,1,2,1,2,1,2,1,2,2,3,2,2,1,2,1,
    1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,
    1,1,2,1,2,1,2,1,2,2,3,2,2,1,2,1,
    1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1
};

namespace NSolver {
    // 座標は 30x30 に境界を追加した 32x32 を 1dim に直した [0,1024) で表す

    using ::operator<<;

    constexpr int N = 32;
    constexpr int NN = N * N;
    constexpr int dir[] = { 1, -N, -1, N };
    int capture_thresh = 15;

    constexpr char board_str[N][N + 1] = {
"################################",
"#X..#..#..#..#..#..#..#..#..#..#",
"#.X#..#..#..#..#..#..#..#..#..##",
"#..X.#..#..#..#..#..#..#..#..#.#",
"##..X..#..#..#..#..#..#..#..#..#",
"#..#.X#..#..#..#..#..#..#..#..##",
"#.#...X.#..#..#..#..#..#..#..#.#",
"##..#..X..#..#..#..#..#..#..#..#",
"#..#..#.X#..#..#..#..#..#..#..##",
"#.#..#...X.#..#..#..#..#..#..#.#",
"##..#..#..X..#..#..#..#..#..#..#",
"#..#..#..#.X#..#..#..#..#..#..##",
"#.#..#..#...X.#..#..#..#..#..#.#",
"##..#..#..#..X..#..#..#..#..#..#",
"#..#..#..#..#.X#..#..#..#..#..##",
"#.#..#..#..#...X.#..#..#..#..#.#",
"##..#..#..#..#@@X..#..#..#..#..#",
"#..#..#..#..#@@#@X#..#..#..#..##",
"#.#..#..#..#@@#@@.X.#..#..#..#.#",
"##..#..#..#@@#@@#..X..#..#..#..#",
"#..#..#..#@@#@@#..#.X#..#..#..##",
"#.#..#..#@@#@@#..#...X.#..#..#.#",
"##..#..#@@#@@#..#..#..X..#..#..#",
"#..#..#@@#@@#..#..#..#.X#..#..##",
"#.#..#@@#@@#..#..#..#...X.#..#.#",
"##..#@@#@@#..#..#..#..#..X..#..#",
"#..#@@#@@#..#..#..#..#..#.X#..##",
"#.#@@#@@#..#..#..#..#..#...X.#.#",
"##@@#@@#..#..#..#..#..#..#..X..#",
"#@@#@@#..#..#..#..#..#..#..#.X.#",
"#@@@@#..#..#..#..#..#..#..#...X#",
"################################"
    };

    struct coord {
        int idx;
        coord(int idx = 0) : idx(idx) {}
        coord(int y, int x) : idx(x | y << 5) {}
        inline int y() const { return idx >> 5; }
        inline int x() const { return idx & 0b11111; }
        inline std::pair<int, int> unpack() const { return { y(), x() }; }
        inline coord& move(int d) { idx += dir[d]; return *this; }
        inline coord moved(int d) const { return coord(*this).move(d); }
        inline int distance(const coord& c) const { return abs(x() - c.x()) + abs(y() - c.y()); }
        inline int get_dir(const coord& to) const {
            if (idx == to.idx) return -1;
            if (y() == to.y()) return x() < to.x() ? 0 : 2;
            if (x() == to.x()) return y() < to.y() ? 3 : 1;
            return -1;
        }
        bool operator==(const coord& c) const { return idx == c.idx; }
        bool operator!=(const coord& c) const { return !(*this == c); }
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

        static Action move(coord crd) { return Action(Type::MOVE, crd.idx); }
        static Action block(int dir) { return Action(Type::BLOCK, dir); }
        static Action wait(int until) { return Action(Type::WAIT, until); }

        inline Type get_type() const { return type; }
        inline coord get_pos() const { return coord(data); }
        inline int get_dir() const { return data; }
        inline int get_time() const { return data; }

        string stringify() const {
            switch (type) {
            case Type::MOVE:  return "move to " + coord(data).stringify();
            case Type::BLOCK: return "block " + d2C[data];
            case Type::WAIT:  return "wait until " + std::to_string(data);
            }
            return "";
        }

    private:

        Type type;
        int data;

        Action(Type type, int data) : type(type), data(data) {}

    };

    struct Human;
    struct Pet;

    struct Task {
        enum struct Type {
            SEQ,    // action 列を逐次的に実行するタスク
            CAP,    // ペット捕獲タスク
        };
        Type type;
        Human* assignee;
        bool is_completed;
    };

    struct SeqTask : Task {
        std::deque<Action> actions;
        SeqTask* next_task = nullptr;
        coord start_pos() const {
            for (const auto& action : actions) if (action.get_type() == Action::Type::MOVE) {
                return action.get_pos();
            }
            assert(false);
            return coord();
        }
        coord end_pos() const {
            for (int i = actions.size() - 1; i >= 0; i--) {
                const auto& action = actions[i];
                if (action.get_type() == Action::Type::MOVE) {
                    return action.get_pos();
                }
            }
            assert(false);
            return coord();
        }
    };

    struct CapTask : Task {
        Pet* target;
    };

    struct Pet {
        enum struct Type { COW, PIG, RABBIT, DOG, CAT };
        int id;
        coord pos;
        Type type;
        bool is_captured;
        CapTask* task;
        Pet(int id = -1, coord pos = -1, Type type = Type(-1)) : id(id), pos(pos), type(type), is_captured(false), task(nullptr) {}
        Pet(int id = -1, coord pos = -1, int type = -1) : id(id), pos(pos), type(Type(type)), is_captured(false), task(nullptr) {}
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
        Human(int id = -1, coord pos = -1) : id(id), pos(pos), task(nullptr) {}
        string stringify() const { return format("Human[%d,%s]", id, pos.stringify().c_str()); }
    };

    struct SeqTaskScheduler {
        // 人の初期位置
        // タスクの開始位置
        // タスクの終了位置
        // 初期盤面における (人 -> タスク開始位置) の各移動コスト
        // 完成盤面における (タスク終了位置 -> タスク開始位置) の各移動コスト
        vector<Human> humans;
        vector<SeqTask> tasks;

        SeqTaskScheduler(const vector<Human>& humans, const vector<SeqTask>& tasks) : humans(humans), tasks(tasks) {}

        vector<vector<int>> run() {
            int nh = humans.size();
            int nt = tasks.size();

            // 初期盤面における (人 -> タスク開始位置) の各移動コスト
            auto idist = make_vector(inf, nh, nt);
            for (int i = 0; i < nh; i++) {
                auto hpos = humans[i].pos;
                for (int j = 0; j < nt; j++) {
                    auto tpos = tasks[j].start_pos();
                    idist[i][j] = hpos.distance(tpos);
                }
            }

            // 完成盤面
            bool blk[NN] = {};
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    if (board_str[y][x] == '#') {
                        blk[y * N + x] = true;
                    }
                }
            }

            // 完成盤面における (タスク終了位置 -> タスク開始位置) の各移動コスト
            auto tdist = make_vector(inf, nt, nt);
            for (int t1 = 0; t1 < nt; t1++) {
                auto endpos = tasks[t1].end_pos();
                int dist[NN];
                Fill(dist, inf);
                std::queue<coord> qu({ endpos });
                dist[endpos.idx] = 0;
                while (!qu.empty()) {
                    auto u = qu.front(); qu.pop();
                    for (int d = 0; d < 4; d++) {
                        auto v = u.moved(d);
                        if (blk[v.idx] || dist[v.idx] != inf) continue;
                        dist[v.idx] = dist[u.idx] + 1;
                        qu.push(v);
                    }
                }
                for (int t2 = 0; t2 < nt; t2++) {
                    auto startpos = tasks[t2].start_pos();
                    tdist[t1][t2] = dist[startpos.idx];
                }
            }

            // 貪欲初期解
            auto get_init_sol = [&]() {
                vector<int> assigned(nt, 0);
                vector<vector<int>> tasklist(nh);
                vector<int> cost(nh, 0);
                while (std::accumulate(assigned.begin(), assigned.end(), 0) != nt) {
                    // 現時点でコスト最小の human に最も近いタスクを貪欲に割当
                    int hid = std::distance(cost.begin(), std::min_element(cost.begin(), cost.end()));
                    if (tasklist[hid].empty()) {
                        int min_dist = inf, min_tid = -1;
                        for (int tid = 0; tid < nt; tid++) if (!assigned[tid]) {
                            if (chmin(min_dist, idist[hid][tid])) {
                                min_tid = tid;
                            }
                        }
                        assigned[min_tid] = 1;
                        tasklist[hid].push_back(min_tid);
                        cost[hid] += min_dist + tasks[min_tid].actions.size();
                    }
                    else {
                        int ptid = tasklist[hid].back();
                        int min_dist = inf, min_tid = -1;
                        for (int tid = 0; tid < nt; tid++) if (!assigned[tid]) {
                            if (chmin(min_dist, tdist[ptid][tid])) {
                                min_tid = tid;
                            }
                        }
                        assigned[min_tid] = 1;
                        tasklist[hid].push_back(min_tid);
                        cost[hid] += min_dist + tasks[min_tid].actions.size();
                    }
                }
                return tasklist;
            };

            auto evaluate = [&](const vector<vector<int>>& sol) {
                vector<int> cost(nh, 0);
                for (int hid = 0; hid < nh; hid++) {
                    const auto& tids = sol[hid];
                    cost[hid] += idist[hid][tids[0]] + tasks[tids[0]].actions.size();
                    for (int i = 1; i < sol[hid].size(); i++) {
                        cost[hid] += tdist[tids[i - 1]][tids[i]] + tasks[tids[i]].actions.size();
                    }
                }
                return *std::max_element(cost.begin(), cost.end());
            };

            auto get_temp = [](double startTemp, double endTemp, double t, double T) {
                return endTemp + (startTemp - endTemp) * (T - t) / T;
            };

            auto best_sol = get_init_sol();
            int min_cost = evaluate(best_sol);
            dump(min_cost);
            constexpr int num_loop = 10000000;
            auto sol = best_sol;
            int cost = min_cost;
            for (int loop = 0; loop < num_loop; loop++) {
                int hid1 = rnd.next_int(nh), hid2 = rnd.next_int(nh);
                int tidx1 = rnd.next_int(sol[hid1].size()), tidx2 = rnd.next_int(sol[hid2].size());
                if (hid1 == hid2 && tidx1 == tidx2) continue;
                std::swap(sol[hid1][tidx1], sol[hid2][tidx2]);
                int now_cost = evaluate(sol);
                int diff = now_cost - cost;
                double temp = get_temp(3.0, 0.0, loop, num_loop);
                double prob = exp(-diff / temp);
                if (rnd.next_double() < prob) {
                    cost = now_cost;
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_sol = sol;
                        dump(min_cost);
                    }
                }
                else {
                    std::swap(sol[hid1][tidx1], sol[hid2][tidx2]);
                }
            }

            return best_sol;
        }
    };

    struct Stats {
        int score = -1;
        int num_humans = 0;
        int num_pets = 0;
        int num_each_pets[5] = {};
        int turn_seq_end = -1;
        int turn_dogkill_start = -1;
        int turn_dogkill_end = -1;
        int turn_all_captured = -1;
        int num_remained = 0;
        int num_each_remained[5] = {};

        void print(std::ostream& out) const {
            out << score << ',' << num_humans << ',' << num_pets;
            for (int ep : num_each_pets) out << ',' << ep;
            out << ',' << turn_seq_end << ',' << turn_dogkill_start << ',' << turn_dogkill_end;
            out << ',' << turn_all_captured << ',' << num_remained;
            for (int er : num_each_remained) out << ',' << er;
            out << '\n';
        }
    };

    struct State {

        std::istream& in;
        std::ostream& out;

        int turn;

        vector<Pet> pets;
        vector<Human> humans;

        bool is_blocked[NN];
        bool is_zone[NN];
        int ctr_human[NN];
        int ctr_pet[NN];

        bool dog_exists;
        bool dog_kill_mode;
        bool dog_kill_completed;

        vector<SeqTask> seq_tasks;

        Stats stats;

        State(std::istream& in, std::ostream& out) : in(in), out(out) {}

        void init() {

            Fill(is_blocked, false);
            for (int y = 0; y < N; y++) is_blocked[y * N] = is_blocked[y * N + N - 1] = true;
            for (int x = 0; x < N; x++) is_blocked[x] = is_blocked[N * (N - 1) + x] = true; // 境界

            Fill(is_zone, false);
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    if (board_str[y][x] == '@') {
                        is_zone[coord(y, x).idx] = true;
                    }
                }
            }

            Fill(ctr_human, 0);
            Fill(ctr_pet, 0);

            turn = 0;

            int num_pets; in >> num_pets;
            for (int pid = 0; pid < num_pets; pid++) {
                coord pos;
                int t;
                cin >> pos >> t;
                t--;
                pets.emplace_back(pid, pos, t);
                ctr_pet[pos.idx]++;
                stats.num_pets++;
                stats.num_each_pets[t]++;
            }

            int num_humans; in >> num_humans;
            for (int hid = 0; hid < num_humans; hid++) {
                coord pos;
                cin >> pos;
                humans.emplace_back(hid, pos);
                ctr_human[pos.idx]++;
                stats.num_humans++;
            }

            dog_exists = false;
            for (const auto& pet : pets) if (pet.type == Pet::Type::DOG) dog_exists = true;
            dog_kill_mode = false;
            dog_kill_completed = false;

            seq_tasks = generate_seq_tasks(); // NOTE: dog_exists に依存あり
            dump(seq_tasks.size());

            for (auto& pet : pets) {
                CapTask* task = new CapTask;
                task->type = Task::Type::CAP;
                task->assignee = nullptr;
                task->is_completed = false;
                task->target = &pet;
                pet.task = task;
            }

        }

        void load_pet_moves() {
            // move
            vector<string> cmd_list(pets.size());
            cin >> cmd_list;
            for (auto& pet : pets) {
                const auto& cmd = cmd_list[pet.id];
                for (char c : cmd) {
                    ctr_pet[pet.pos.idx]--;
                    pet.pos.move(c2d[c]);
                    ctr_pet[pet.pos.idx]++;
                }
            }
        }

        void update_cap_task_status() {

            // 捕獲判定
            UnionFind tree(NN);
            // yoko
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N - 1; x++) {
                    if (is_blocked[y * N + x] || is_blocked[y * N + x + 1]) continue;
                    tree.unite(y * N + x, y * N + x + 1);
                }
            }
            // tate
            for (int y = 0; y < N - 1; y++) {
                for (int x = 0; x < N; x++) {
                    if (is_blocked[y * N + x] || is_blocked[(y + 1) * N + x]) continue;
                    tree.unite(y * N + x, (y + 1) * N + x);
                }
            }

            for (auto& pet : pets) if (!pet.is_captured) {
                if (tree.size(pet.pos.idx) <= capture_thresh) {
                    pet.is_captured = true;
                    pet.task->is_completed = true;
                    dump("captured!", turn, (pet.task->assignee ? pet.task->assignee->stringify() : "null"), pet, tree.size(pet.pos.idx));
                    if (pet.task->assignee) {
                        pet.task->assignee->task = nullptr;
                        pet.task->assignee = nullptr;
                    }
                }
            }

        }

        bool can_block(coord pos) const {
            // 重複して設置しない || 人間に刺さない || ペットに刺さない
            int idx = pos.idx;
            if (is_blocked[idx] || ctr_human[idx] || ctr_pet[idx]) return false;
            for (int d = 0; d < 4; d++) {
                int nidx = idx + dir[d];
                if (ctr_pet[nidx]) return false; // ペットの 4 近傍に置かない
            }
            return true;
        }

        // from から block を避けての最短距離
        vector<int> bfs(const coord& from) const {
            assert(!is_blocked[from.idx]);
            vector<int> dist(NN, inf);
            std::queue<coord> qu({ from });
            dist[from.idx] = 0;
            while (!qu.empty()) {
                auto u = qu.front(); qu.pop();
                for (int d = 0; d < 4; d++) {
                    auto v = u.moved(d);
                    if (is_blocked[v.idx] || dist[v.idx] != inf) continue;
                    dist[v.idx] = dist[u.idx] + 1;
                    qu.push(v);
                }
            }
            return dist;
        }

        // 人間が target に最短経路で移動（候補複数ならランダム）
        // 柵の設置は先に処理されている
        char resolve_move(Human& human, const Action& action) {
            assert(action.get_type() == Action::Type::MOVE);

            auto spos = human.pos;
            auto gpos = action.get_pos();

            assert(!is_blocked[gpos.idx]);
            assert(spos != gpos);

            auto dist = bfs(gpos);
            assert(dist[spos.idx] != inf);

            // 最短経路移動方向の候補を調べる
            int min_dist = inf;
            vector<int> cands;
            for (int d = 0; d < 4; d++) {
                auto npos = spos.moved(d);
                // 柵を踏まない
                if (is_blocked[npos.idx] || dist[npos.idx] > min_dist) continue;
                if (dist[npos.idx] < min_dist) {
                    min_dist = dist[npos.idx];
                    cands.clear();
                }
                cands.push_back(d);
            }

            assert(min_dist != inf);

            // ランダムで選ぶ
            int d = cands[rnd.next_int(cands.size())];

            // 移動
            ctr_human[human.pos.idx]--;
            human.pos.move(d);
            ctr_human[human.pos.idx]++;

            //show();

            return d2C[d];
        }

        void update_seq_task_status() {
            for (auto& human : humans) {
                bool updated = true;
                while (updated) {
                    updated = false;
                    if (!human.task || human.task->type != Task::Type::SEQ) break;
                    auto stask = reinterpret_cast<SeqTask*>(human.task);
                    if (stask->actions.empty()) {
                        stask->is_completed = true;
                        stask->assignee = nullptr;
                        human.task = stask->next_task;
                        updated = true;
                        dump(turn, "seq task completed!", human);
                        continue;
                    }
                    const auto& action = stask->actions.front();
                    auto atype = action.get_type();
                    if (atype == Action::Type::MOVE) {
                        auto to = action.get_pos();
                        auto dist = bfs(human.pos);
                        if (dist[to.idx] == inf) {
                            // 到達不能
                            stask->actions.clear();
                            stask->is_completed = true;
                            stask->assignee = nullptr;
                            human.task = stask->next_task;
                            updated = true;
                            dump(turn, "seq task aborted!", human);
                            continue;
                        }
                        if (human.pos == to) {
                            stask->actions.pop_front();
                            updated = true;
                            continue;
                        }
                    }
                    else if (atype == Action::Type::BLOCK) {
                        int d = action.get_dir();
                        if (is_blocked[human.pos.moved(d).idx]) {
                            stask->actions.pop_front();
                            updated = true;
                            continue;
                        }
                    }
                }
            }
        }

        // 柵の設置
        char resolve_block(const Human& human, const Action& action) {
            assert(action.get_type() == Action::Type::BLOCK);
            int d = action.get_dir();
            auto pos = human.pos.moved(d);
            if (can_block(pos)) {
                is_blocked[pos.idx] = true;
                // block を置いたことで生じる影響の解決
                update_seq_task_status();
                update_cap_task_status();
                //show();
                return d2c[d];
            }
            update_seq_task_status();
            return '.';
        }

        // (hy, hx) にいる人間がマス (by, bx) に柵を置いた時、ペットを面積 thresh 以下の領域に収容できるか？
        // 人間が trap 側領域に入るのは許容しない
        bool can_trap(const Human& human, int d, int thresh) {
            auto bpos = human.pos.moved(d);
            if (!can_block(bpos)) return false;
            is_blocked[bpos.idx] = true;

            UnionFind tree(NN);
            // yoko
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N - 1; x++) {
                    if (is_blocked[y * N + x] || is_blocked[y * N + x + 1]) continue;
                    tree.unite(y * N + x, y * N + x + 1);
                }
            }
            // tate
            for (int y = 0; y < N - 1; y++) {
                for (int x = 0; x < N; x++) {
                    if (is_blocked[y * N + x] || is_blocked[(y + 1) * N + x]) continue;
                    tree.unite(y * N + x, (y + 1) * N + x);
                }
            }

            for (const auto& h : humans) {
                if (tree.size(h.pos.idx) <= thresh) {
                    // 人間が狭い領域に入ってしまう
                    is_blocked[bpos.idx] = false;
                    return false;
                }
            }

            for (int d = 0; d < 4; d++) {
                coord npos = bpos.moved(d);
                if (is_blocked[npos.idx]) continue;
                int area = tree.size(npos.idx);
                if (area > thresh) continue;
                int nbid = tree.find(npos.idx);
                // pet
                for (const auto& pet : pets) {
                    if (pet.is_captured) continue;
                    if (tree.find(pet.pos.idx) == nbid) {
                        is_blocked[bpos.idx] = false;
                        return true;
                    }
                }
            }

            is_blocked[bpos.idx] = false;
            return false;
        }

        void insert_interrupt_task() {
            constexpr int dist_thresh = 2;
            vector<std::pair<double, coord>> spots;
            for (int y = 1; y < N - 1; y++) {
                for (int x = 1; x < N - 1; x++) {
                    double score = evaluate(coord(y, x));
                    if (score > 0.0) {
                        spots.emplace_back(score, coord(y, x));
                    }
                }
            }
            std::sort(spots.begin(), spots.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
                });
            dump(spots);
            // 近い spot があれば "spot まで移動するタスク", "spot から元の位置に戻るタスク" を挿入
            for (auto& human : humans) {
                if (human.task && human.task->type == Task::Type::CAP) continue;
                for (auto [score, spot] : spots) {
                    if (human.pos.distance(spot) <= dist_thresh) {
                        SeqTask* task1 = new SeqTask();
                        task1->assignee = &human;
                        task1->is_completed = false;
                        task1->type = Task::Type::SEQ;
                        task1->actions.push_back(Action::move(spot));
                        SeqTask* task2 = new SeqTask();
                        task2->assignee = &human;
                        task2->is_completed = false;
                        task2->type = Task::Type::SEQ;
                        task2->actions.push_back(Action::move(human.pos));
                        task1->next_task = task2;
                        if (human.task) task2->next_task = reinterpret_cast<SeqTask*>(human.task);
                        human.task = task1;
                    }
                }
            }
            update_seq_task_status();
        }

        string resolve_actions() {

            string actions(humans.size(), '.');

            if (all_cap_task_completed()) return actions;

            insert_interrupt_task();

            // 1. ペット封印を解決
            for (auto& human : humans) {
                for (int d = 0; d < 4; d++) {
                    if (can_trap(human, d, capture_thresh)) {
                        actions[human.id] = resolve_block(human, Action::block(d));
                        break;
                    }
                }
            }

            // 2. seq 柵の設置を解決
            for (auto& human : humans) {
                if (actions[human.id] != '.' || !human.task || human.task->type != Task::Type::SEQ) continue;
                auto stask = reinterpret_cast<SeqTask*>(human.task);
                if (stask->actions.front().get_type() != Action::Type::BLOCK) continue;
                actions[human.id] = resolve_block(human, stask->actions.front());
            }

            // 3. seq 移動を解決
            for (auto& human : humans) {
                if (actions[human.id] != '.' || !human.task || human.task->type != Task::Type::SEQ) continue;
                auto stask = reinterpret_cast<SeqTask*>(human.task);
                if (stask->actions.front().get_type() != Action::Type::MOVE) continue;
                actions[human.id] = resolve_move(human, stask->actions.front());
            }

            // 4. cap task
            for (auto& human : humans) {
                if (actions[human.id] != '.' || !human.task || human.task->type != Task::Type::CAP) continue;
                auto ctask = reinterpret_cast<CapTask*>(human.task);
                if (human.pos.distance(ctask->target->pos) <= 2) {
                    // 距離 2 まで接近（1 以下だと cow を捕獲できない)
                    vector<coord> cands;
                    for (int d = 0; d < 4; d++) {
                        auto npos = human.pos.moved(d);
                        if (!is_blocked[npos.idx]) {
                            cands.push_back(npos);
                        }
                    }
                    if (!cands.empty()) {
                        auto act = Action::move(cands[rnd.next_int(cands.size())]);
                        actions[human.id] = resolve_move(human, act);
                    }
                }
                else {
                    auto act = Action::move(ctask->target->pos);
                    actions[human.id] = resolve_move(human, act);
                }
            }

            // 6. gather
            for (auto& human : humans) {
                if (!human.task && actions[human.id] == '.' && human.pos != coord(16, 16)) {
                    auto act = Action::move(coord(16, 16));
                    actions[human.id] = resolve_move(human, act);
                }
            }

            return actions;
        }

        bool all_seq_task_completed() const {
            for (const auto& task : seq_tasks) if (!task.is_completed) return false;
            return true;
        }

        bool all_cap_task_completed() const {
            for (const auto& pet : pets) if (!pet.task->is_completed) return false;
            return true;
        }

        void assign_tasks() {

            // rearrange capture task

            // cancel all capture task
            for (auto& pet : pets) {
                if (pet.type == Pet::Type::DOG || pet.is_captured || !pet.task->assignee) continue;
                auto& human = *pet.task->assignee;
                human.task = nullptr;
                pet.task->assignee = nullptr;
            }

            // assign capture task
            for (auto& pet : pets) if (pet.type != Pet::Type::DOG && !pet.is_captured && !pet.task->assignee && !is_zone[pet.pos.idx]) {
                // 最寄りの暇人にタスクをアサイン
                int min_dist = inf;
                Human* assigned_human = nullptr;
                auto dist = bfs(pet.pos);
                for (auto& human : humans) if (!human.task) {
                    if (chmin(min_dist, dist[human.pos.idx])) assigned_human = &human;
                }
                if (assigned_human) {
                    pet.task->assignee = assigned_human;
                    assigned_human->task = pet.task;
                }
            }

        }

        // 初期位置とコマンド列から action list を生成
        SeqTask generate_seq_task(coord pos, const string& cmd_list) {

            SeqTask task;
            task.type = Task::Type::SEQ;
            task.assignee = nullptr;
            task.is_completed = false;

            auto& actions = task.actions;
            actions.push_back(Action::move(pos));
            for (char c : cmd_list) {
                assert(c != '.'); // wait は許容しない
                if (isupper(c)) { // move
                    pos.move(c2d[c]);
                    actions.push_back(Action::move(pos));
                }
                else { // block
                    actions.push_back(Action::block(c2d[c]));
                }
            }

            return task;
        }

        vector<SeqTask> generate_seq_tasks() {

            vector<SeqTask> tasks;

            auto rep = [](int n, const string& s) {
                string res;
                while (n--) res += s;
                return res;
            };

            if (dog_exists) {
                {
                    string cmd = "rdLuDdLuDdLuDdLuDdLuDdLuDdLuDdLuDdLuDdLuDdLuDdLuDLuD";
                    cmd += "RRRRR" + rep(11, "lRrU") + "l";
                    tasks.push_back(generate_seq_task({ 17, 14 }, cmd));
                }

                for (int k = 0; k < 4; k++) {
                    tasks.push_back(generate_seq_task({ 6 * k + 6 , 1 }, rep(3 * k + 1, "dUuR") + "dr"));
                }
                for (int k = 0; k < 5; k++) {
                    tasks.push_back(generate_seq_task({ 1, 6 * k + 6 }, rep(3 * k + 2, "rLlD") + "r"));
                }
                for (int k = 0; k < 3; k++) {
                    tasks.push_back(generate_seq_task({ 30, 24 - 6 * k }, rep(3 * k + 2, "lRrU") + "l"));
                }
                for (int k = 0; k < 4; k++) {
                    tasks.push_back(generate_seq_task({ 24 - 6 * k, 30 }, rep(3 * k + 2, "uDdL") + "ul"));
                }
            }
            else {
                for (int k = 0; k < 5; k++) {
                    tasks.push_back(generate_seq_task({ 6 * k + 6 , 1 }, rep(3 * k + 1, "dUuR") + "dr"));
                }
                for (int k = 0; k < 5; k++) {
                    tasks.push_back(generate_seq_task({ 1, 6 * k + 6 }, rep(3 * k + 2, "rLlD") + "r"));
                }
                for (int k = 0; k < 4; k++) {
                    tasks.push_back(generate_seq_task({ 30, 24 - 6 * k }, rep(3 * k + 2, "lRrU") + "l"));
                }
                for (int k = 0; k < 4; k++) {
                    tasks.push_back(generate_seq_task({ 24 - 6 * k, 30 }, rep(3 * k + 2, "uDdL") + "ul"));
                }
            }

            return tasks;
        }

        bool all_dog_captured() const {
            for (const auto& pet : pets) if (pet.type == Pet::Type::DOG && !pet.is_captured) return false;
            return true;
        }

        void toggle_dog_kill_mode() {
            if (!dog_exists) return;
            if (dog_kill_completed) return;
            if (!dog_kill_mode) {
                // 全 seq task が終了している
                if (!all_seq_task_completed()) return;
                // この時点で犬が全部捕獲されていたら (30,2) に block を置くタスクを発行
                // (30,2) に最も近い人間の捕獲タスクを（あれば）キャンセルして割当て
                if (all_dog_captured()) {
                    SeqTask task;
                    task.actions.push_back(Action::move(coord(30, 1)));
                    task.actions.push_back(Action::block(0));
                    task.type = Task::Type::SEQ;
                    task.assignee = nullptr;
                    task.is_completed = false;
                    task.next_task = nullptr;
                    seq_tasks.push_back(task);
                    dog_kill_completed = true;
                    stats.turn_dogkill_end = turn;
                    dump(turn, "all dogs are already captured!");
                    Human* nearest = &humans[0];
                    {
                        auto dist = bfs(coord(30, 2));
                        int min_dist = dist[humans[0].pos.idx];
                        for (int hid = 1; hid < humans.size(); hid++) {
                            int d = dist[humans[hid].pos.idx];
                            if (d < min_dist) {
                                nearest = &humans[hid];
                                min_dist = d;
                            }
                        }
                    }
                    if (nearest->task) {
                        if (nearest->task->type == Task::Type::SEQ) {
                            auto stask = reinterpret_cast<SeqTask*>(nearest->task);
                            while (stask->next_task != nullptr) {
                                stask = stask->next_task;
                            }
                            stask->next_task = &seq_tasks.back();
                        }
                        else {
                            nearest->task->assignee = nullptr;
                            nearest->task = &seq_tasks.back();
                        }
                    }
                    else {
                        nearest->task = &seq_tasks.back();
                    }
                    return;
                }
                dog_kill_mode = true;
                dump(turn, "dog kill mode start!");
                stats.turn_dogkill_start = turn;
            }
            else {
                // 犬を全捕獲したらオフ
                for (const auto& pet : pets) {
                    if (pet.type == Pet::Type::DOG && !pet.is_captured) {
                        return;
                    }
                }
                dog_kill_mode = false;
                dog_kill_completed = true;
                dump(turn, "dog kill mode end!");
                stats.turn_dogkill_end = turn;
            }
        }

        string resolve_dogkill_actions() {

            auto pos1 = coord(17, 13);
            auto pos2 = coord(27, 3);
            auto b1 = coord(18, 13);
            auto b2 = coord(27, 4);

            auto all_moved = [&]() {
                for (int i = 0; i < humans.size() - 1; i++) {
                    if (humans[i].pos != pos1) return false;
                }
                if (humans.back().pos != pos2) return false;
                return true;
            };

            auto all_capture = [&]() {
                if (!can_block(b1) || !can_block(b2)) return false;
                is_blocked[b1.idx] = true;
                is_blocked[b2.idx] = true;
                auto dist = bfs(b1.moved(2));
                for (const auto& pet : pets) if (!pet.is_captured && pet.type == Pet::Type::DOG && dist[pet.pos.idx] == inf) {
                    is_blocked[b1.idx] = false;
                    is_blocked[b2.idx] = false;
                    return false;
                }
                is_blocked[b1.idx] = false;
                is_blocked[b2.idx] = false;
                return true;
            };

            string actions(humans.size(), '.');

            if (all_moved() && all_capture()) {
                actions[0] = resolve_block(humans[0], Action::block(3));
                actions.back() = resolve_block(humans.back(), Action::block(0));
                return actions;
            }

            for (int i = 0; i < humans.size() - 1; i++) {
                auto& human = humans[i];
                if (human.pos != pos1) {
                    actions[human.id] = resolve_move(human, Action::move(pos1));
                }
            }
            {
                auto& human = humans.back();
                if (human.pos != pos2) {
                    actions[human.id] = resolve_move(human, Action::move(pos2));
                }
            }

            return actions;
        }

        int calc_score() const {
            UnionFind tree(NN);
            // yoko
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N - 1; x++) {
                    if (is_blocked[y * N + x] || is_blocked[y * N + x + 1]) continue;
                    tree.unite(y * N + x, y * N + x + 1);
                }
            }
            // tate
            for (int y = 0; y < N - 1; y++) {
                for (int x = 0; x < N; x++) {
                    if (is_blocked[y * N + x] || is_blocked[(y + 1) * N + x]) continue;
                    tree.unite(y * N + x, (y + 1) * N + x);
                }
            }

            double score = 0.0;

            std::map<int, int> pet_in_region;
            for (const auto& pet : pets) pet_in_region[tree.find(pet.pos.idx)]++;

            for (const auto& human : humans) {
                int r = tree.find(human.pos.idx);
                score += tree.size(r) * pow(2.0, -pet_in_region[r]) / 900.0;
            }

            return (int)round(1e8 * score / humans.size());
        }

        void update_stats() {
            stats.score = calc_score();
            if (stats.turn_seq_end == -1 && all_seq_task_completed()) stats.turn_seq_end = turn;
            if (stats.turn_all_captured == -1 && all_cap_task_completed()) stats.turn_all_captured = turn;
        }

        void summary_stats() {
            stats.score = calc_score();
            for (const auto& pet : pets) if (!pet.is_captured) {
                stats.num_remained++;
                stats.num_each_remained[(int)pet.type]++;
            }
        }

        void solve() {

            init();

            dump(humans.size(), pets.size());

            show();

            { // seq task assign
                auto assign = SeqTaskScheduler(humans, seq_tasks).run();
                for (int hid = 0; hid < humans.size(); hid++) {
                    auto& human = humans[hid];
                    SeqTask* task = &seq_tasks[assign[hid][0]];
                    human.task = task;
                    for (int i = 1; i < assign[hid].size(); i++) {
                        task->next_task = &seq_tasks[assign[hid][i]];
                        task = task->next_task;
                    }
                }
            }

            update_seq_task_status();
            update_stats();
            while (turn < MAX_TURN) {
                if (all_seq_task_completed()) capture_thresh = 35;
                toggle_dog_kill_mode();
                string actions;
                if (dog_kill_mode) actions = resolve_dogkill_actions();
                else actions = resolve_actions();
                cout << actions << endl;
                load_pet_moves();
                update_seq_task_status();
                assign_tasks();

                turn++;

                update_stats();
                show();
            }

            summary_stats();

            dump(stats.score);

#ifdef ENABLE_STATS_DUMP
            std::ofstream ofs("stats.csv", std::ios::app);
            stats.print(ofs);
#endif

            for (const auto& pet : pets) if (!pet.is_captured) {
                dump(pet);
            }

        }

        double evaluate(const coord& pos) {
            static constexpr int d8[] = { 1, -31, -32, -33, -1, 31, 32, 33 };

            if (is_blocked[pos.idx] || ctr_pet[pos.idx]) return 0.0;
            for (int d = 0; d < 4; d++) if (ctr_pet[pos.moved(d).idx]) return 0.0;

            int mask = 0;
            for (int d = 0; d < 8; d++) {
                mask |= (is_blocked[pos.idx + d8[d]]) ? (1 << d) : 0;
            }
            if (cc_table[mask] < 2) return 0.0;

            // (num pets) / (area)
            is_blocked[pos.idx] = true;
            UnionFind tree(NN);
            // yoko
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N - 1; x++) {
                    if (is_blocked[y * N + x] || is_blocked[y * N + x + 1]) continue;
                    tree.unite(y * N + x, y * N + x + 1);
                }
            }
            // tate
            for (int y = 0; y < N - 1; y++) {
                for (int x = 0; x < N; x++) {
                    if (is_blocked[y * N + x] || is_blocked[(y + 1) * N + x]) continue;
                    tree.unite(y * N + x, (y + 1) * N + x);
                }
            }
            is_blocked[pos.idx] = false;

            double score = 0.0;
            for (const auto& pet : pets) {
                if (pet.is_captured) continue;
                int sz = tree.size(pet.pos.idx);
                if (sz <= capture_thresh) {
                    score += 1.0 / sz;
                }
            }

            return score;
        }

#ifdef HAVE_OPENCV_HIGHGUI
        static cv::Mat_<cv::Vec3b> get_empty_icon(const cv::Size& size, const cv::Vec3b bgcolor = cv::Vec3b(200, 200, 200)) {
            cv::Mat_<cv::Vec3b> img(size.height, size.height, bgcolor);
            cv::rectangle(img, cv::Rect(0, 0, size.width, size.height), cv::Scalar(255, 255, 255));
            return img;
        }

        static cv::Mat_<cv::Vec3b> load_icon(
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

        static constexpr int icon_size = 32;
        const cv::Size img_size = cv::Size(icon_size, icon_size);
        const cv::Mat_<cv::Vec3b> icon_empty = get_empty_icon(img_size);
        const cv::Mat_<cv::Vec3b> icon_human = load_icon("img/human.png", img_size);
        const cv::Mat_<cv::Vec3b> icon_block = load_icon("img/block.png", img_size);
        const cv::Mat_<cv::Vec3b> icon_cow = load_icon("img/cow.png", img_size);
        const cv::Mat_<cv::Vec3b> icon_pig = load_icon("img/pig.png", img_size);
        const cv::Mat_<cv::Vec3b> icon_rabbit = load_icon("img/rabbit.png", img_size);
        const cv::Mat_<cv::Vec3b> icon_dog = load_icon("img/dog.png", img_size);
        const cv::Mat_<cv::Vec3b> icon_cat = load_icon("img/cat.png", img_size);
        const cv::Mat_<cv::Vec3b> icon_pets[5] = { icon_cow, icon_pig, icon_rabbit, icon_dog, icon_cat };
        const cv::Mat_<cv::Vec3b> icon_all[8] = { icon_cow, icon_pig, icon_rabbit, icon_dog, icon_cat, icon_human, icon_block, icon_empty };

        void show(int delay = 0) {

            auto get_roi = [&](int y, int x) {
                return cv::Rect(x * icon_size, y * icon_size, icon_size, icon_size);
            };

            auto get_point = [&](coord pos) {
                return cv::Point(pos.x() * icon_size + icon_size / 2, pos.y() * icon_size + icon_size / 2);
            };

            cv::Mat_<cv::Vec3b> img(N * icon_size, N * icon_size, cv::Vec3b(0, 0, 0));
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    (is_blocked[coord(y, x).idx] ? icon_block : icon_empty).copyTo(img(get_roi(y, x)));
                }
            }
            for (const auto& pet : pets) {
                auto [y, x] = pet.pos.unpack();
                cv::Rect roi(x * icon_size, y * icon_size, icon_size, icon_size);
                icon_pets[(int)pet.type].copyTo(img(get_roi(y, x)));
            }
            for (const auto& human : humans) {
                auto [y, x] = human.pos.unpack();
                icon_human.copyTo(img(get_roi(y, x)));
                if (!dog_kill_mode && human.task && human.task->type == Task::Type::CAP) {
                    auto ctask = reinterpret_cast<CapTask*>(human.task);
                    auto ppos = ctask->target->pos;
                    cv::arrowedLine(img, get_point(human.pos), get_point(ppos), cv::Scalar(0, 0, 0));
                }
            }
            for (int y = 0; y < N; y++) {
                for (int x = 0; x < N; x++) {
                    cv::Rect roi(x * icon_size, y * icon_size, icon_size, icon_size);
                    double score = evaluate(coord(y, x));
                    int val = (int)round(std::min(255.0 * score * 5, 255.0));
                    if (val) {
                        cv::Scalar blue(255, 0, 0), red(0, 0, 255);
                        double ratio = (double)val / 255;
                        cv::rectangle(img, roi, blue * ratio + red * (1.0 - ratio), 2);
                    }
                }
            }

            cv::putText(img, format("turn: %d", turn), cv::Point(icon_size * 2 / 3, icon_size * 2 / 3), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3);
            cv::putText(img, format("turn: %d", turn), cv::Point(icon_size * 2 / 3, icon_size * 2 / 3), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

            cv::imshow("vis", img);
            cv::imwrite(format("C:\\Users\\komori3\\OneDrive\\dev\\compro\\heuristic\\tasks\\AHC008\\tmp\\%3d.png", turn), img);
            cv::waitKey(delay);
        }
#else
        void show(int delay = 0) const {}
#endif

    };

}

#ifdef _MSC_VER
void cmd_generator() {
    constexpr int dy[] = { 0, -1, 0, 1 };
    constexpr int dx[] = { 1, 0, -1, 0 };
    int c2d[256];
    c2d['r'] = c2d['R'] = 0;
    c2d['u'] = c2d['U'] = 1;
    c2d['l'] = c2d['L'] = 2;
    c2d['d'] = c2d['D'] = 3;
    string supper = "RULD";
    string slower = "ruld";

    char board[32][33] = {};
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 32; x++) {
            board[y][x] = '#';
        }
    }
    for (int y = 1; y <= 30; y++) {
        for (int x = 1; x <= 30; x++) {
            board[y][x] = '.';
        }
    }

    string cmd;

    int sy, sx;
    int y, x;
    cerr << "position: ";
    cin >> y >> x;
    sy = y; sx = x;

    auto print = [&]() {
        std::ostringstream oss;
        oss << "tasks.push_back(generate_seq_task(coord(" << sy << ',' << sx << "), \"" << cmd << "\"));\n";
        for (const auto& line : board) oss << line << '\n';
        cerr << oss.str();
    };

    board[y][x] = '*';

    while (true) {
        print();
        char c = getch();
        if (c == 13) break;
        if (c == 8) {
            if (!cmd.empty()) {
                char pc = cmd.back(); cmd.pop_back();
                int d = c2d[pc];
                if (supper.find(pc) != string::npos) {
                    board[y][x] = '.';
                    x -= dx[d];
                    y -= dy[d];
                    board[y][x] = '*';
                }
                else if (slower.find(pc) != string::npos) {
                    board[y + dy[d]][x + dx[d]] = '.';
                }
            }
        }
        else if (c == 'x') {
            sy = y; sx = x;
            for (int y = 1; y <= 30; y++) {
                for (int x = 1; x <= 30; x++) {
                    board[y][x] = '.';
                }
            }
            board[y][x] = '*';
            cmd.clear();
        }
        else if (supper.find(c) != string::npos) {
            int d = c2d[c];
            if (board[y + dy[d]][x + dx[d]] == '.') {
                board[y][x] = '.';
                y += dy[d];
                x += dx[d];
                board[y][x] = '*';
                cmd += c;
            }
        }
        else if (slower.find(c) != string::npos) {
            int d = c2d[c];
            if (board[y + dy[d]][x + dx[d]] == '.') {
                board[y + dy[d]][x + dx[d]] = '#';
                cmd += c;
            }
        }
    }

    exit(1);
}
#endif

int main() {

    //cmd_generator();

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    c2d['R'] = c2d['r'] = 0;
    c2d['U'] = c2d['u'] = 1;
    c2d['L'] = c2d['l'] = 2;
    c2d['D'] = c2d['d'] = 3;

    NSolver::State state(cin, cout);
    state.solve();

    //cerr << timer.elapsed_ms() << endl;

    return 0;
}