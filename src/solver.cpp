#include <bits/stdc++.h>
#include <random>
#ifdef _MSC_VER
#define ENABLE_VIS
#define ENABLE_DUMP
#endif
#ifdef _MSC_VER
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


constexpr int MAX_TURN = 300;
constexpr char d2C[5] = "RULD";
constexpr char d2c[5] = "ruld";
int c2d[256];
constexpr int inf = INT_MAX / 8;

namespace NSolver {
    // 座標は 30x30 に境界を追加した 32x32 を 1dim に直した [0,1024) で表す

    using ::operator<<;

    constexpr int N = 32;
    constexpr int NN = N * N;
    constexpr int dir[] = { 1, -N, -1, N };
    constexpr int capture_thresh = 35;

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
        inline coord get_coord() const { return coord(data); }
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
            DOG     // 犬を肉にするタスク
        };
        Type type;
        Human* assignee;
        bool is_cancelable; // sequential は不可
        bool is_completed;
    };

    struct SeqTask : Task {
        std::deque<Action> actions;
    };

    struct CapTask : Task {
        Pet* target;
    };

    struct DogTask : Task {
        coord from; // キルゾーン入り口
        coord to;   // キルゾーン袋小路
    };

    struct TaskGenerator {

        // 初期位置とコマンド列から action list を生成
        SeqTask generate_sequential_task(coord pos, const string& cmd_list) {

            SeqTask task;
            task.type = Task::Type::SEQ;
            task.assignee = nullptr;
            task.is_cancelable = false;
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

        vector<SeqTask> generate_sequential_tasks() {

            vector<SeqTask> tasks;

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
        //std::deque<Action> qu;
        Human(int id = -1, coord pos = -1) : id(id), pos(pos), task(nullptr) {}
        void assign(SeqTask* task_) {
            this->task = task_;
            task->assignee = this;
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

        vector<SeqTask> seq_tasks;
        vector<CapTask> cap_tasks;

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
                t--;
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
            seq_tasks = TaskGenerator().generate_sequential_tasks();
            cap_tasks.resize(num_pets);
            for (auto& pet : pets) {
                CapTask task;
                task.type = Task::Type::CAP;
                task.assignee = nullptr;
                task.is_cancelable = true;
                task.is_completed = false;
                task.target = &pet;
                cap_tasks[pet.id] = task;
                pet.task = &cap_tasks[pet.id];
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

        void update_pet_status() {

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
                    dump("captured!", turn, (pet.task->assignee ? pet.task->assignee->stringify() : "null") , pet, tree.size(pet.pos.idx));
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

        // 人間が target に最短経路で移動（候補複数ならランダム）
        // 柵の設置は先に処理されている
        char resolve_move(Human& human, const Action& action) {
            static int dist[NN];

            assert(action.get_type() == Action::Type::MOVE);

            int sidx = human.pos.idx, gidx = action.get_coord().idx;

            assert(sidx != gidx);
            
            // 目的地からの距離を計算
            Fill(dist, inf);
            std::queue<int> qu;
            qu.emplace(gidx);
            dist[gidx] = 0;
            while (!qu.empty()) {
                int idx = qu.front(); qu.pop();
                for (int d = 0; d < 4; d++) {
                    int nidx = idx + dir[d];
                    if (is_blocked[nidx] || dist[nidx] != inf) continue;
                    dist[nidx] = dist[idx] + 1;
                    qu.emplace(nidx);
                }
            }

            assert(dist[sidx] != inf);

            // 最短経路移動方向の候補を調べる
            int min_dist = inf;
            vector<int> cands;
            for (int d = 0; d < 4; d++) {
                int nidx = sidx + dir[d];
                // 柵を踏まない
                if (is_blocked[nidx] || dist[nidx] > min_dist) continue;
                if (dist[nidx] < min_dist) {
                    min_dist = dist[nidx];
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

        // 柵の設置
        char resolve_block(const Human& human, const Action& action) {
            assert(action.get_type() == Action::Type::BLOCK);
            int d = action.get_dir();
            auto pos = human.pos.moved(d);
            if (can_block(pos)) {
                is_blocked[pos.idx] = true;
                update_pet_status();
                //show();
                return d2c[d];
            }
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

        string resolve_actions() {

            string actions(humans.size(), '.');

            // 1. ペット封印を解決
            for (auto& human : humans) {
                for (int d = 0; d < 4; d++) {
                    if (can_trap(human, d, capture_thresh)) {
                        actions[human.id] = resolve_block(human, Action::block(d));
                        break;
                    }
                }
            }

            // 2. sequential 柵の設置を解決
            for (auto& human : humans) {
                if (actions[human.id] != '.' || !human.task || human.task->type != Task::Type::SEQ) continue;
                auto stask = reinterpret_cast<SeqTask*>(human.task);
                if (stask->actions.front().get_type() != Action::Type::BLOCK) continue;
                actions[human.id] = resolve_block(human, stask->actions.front());
            }

            // 3. sequential 移動を解決
            for (auto& human : humans) {
                if (actions[human.id] != '.' || !human.task || human.task->type != Task::Type::SEQ) continue;
                auto stask = reinterpret_cast<SeqTask*>(human.task);
                if (stask->actions.front().get_type() != Action::Type::MOVE) continue;
                actions[human.id] = resolve_move(human, stask->actions.front());
            }

            // 4. capture task
            for (auto& human : humans) {
                if (human.task && human.task->type == Task::Type::CAP && actions[human.id] == '.') {
                    auto ctask = reinterpret_cast<CapTask*>(human.task);
                    if (human.pos.distance(ctask->target->pos) <= 2) {
                        // 距離 2 まで接近（1 以下だと cow を捕獲できない)
                        auto act = Action::move(human.pos.moved(rnd.next_int(4)));
                        actions[human.id] = resolve_move(human, act);
                    }
                    else {
                        auto act = Action::move(ctask->target->pos);
                        actions[human.id] = resolve_move(human, act);
                    }
                }
            }

            // 5. gather
            for (auto& human : humans) {
                if (!human.task && actions[human.id] == '.' && human.pos != coord(16, 16)) {
                    auto act = Action::move(coord(16, 16));
                    actions[human.id] = resolve_move(human, act);
                }
            }

            // 6. dog task

            return actions;
        }

        void update_queue(Human& human) {
            auto& [id, pos, task] = human;
            if (!task || task->type != Task::Type::SEQ) return;
            auto stask = reinterpret_cast<SeqTask*>(task);
            auto& qu = stask->actions;
            while (!qu.empty()) {
                bool updated = false;
                const auto& act = qu.front();
                auto type = act.get_type();
                switch (type) {
                case Action::Type::MOVE:
                {
                    if (pos == act.get_coord()) {
                        qu.pop_front();
                        updated = true;
                    }
                    break;
                }
                case Action::Type::BLOCK:
                {
                    if (is_blocked[pos.moved(act.get_dir()).idx]) {
                        qu.pop_front();
                        updated = true;
                    }
                    break;
                }
                case Action::Type::WAIT:
                {
                    assert(false);
                    break;
                }
                }
                if (!updated) break;
            }
            if (qu.empty()) {
                stask->assignee = nullptr;
                stask->is_completed = true;
                human.task = nullptr;
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

        void assign_tasks() {
            // assign seqential task
            for (auto& human : humans) if (!human.task) {
                int min_dist = inf;
                SeqTask* selected_task = nullptr;
                auto from = human.pos;
                for (auto& task : seq_tasks) if (!task.assignee && !task.is_completed) {
                    auto to = task.actions.front().get_coord();
                    int dist = calc_dist(from, to);
                    if (chmin(min_dist, dist)) selected_task = &task;
                }
                if (selected_task) {
                    human.assign(selected_task);
                }
            }
            // assign capture task
            for (auto& pet : pets) if (pet.type != Pet::Type::DOG && !pet.is_captured && !pet.task->assignee) {
                // 最寄りの暇人にタスクをアサイン
                int min_dist = inf;
                Human* assigned_human = nullptr;
                for (auto& human : humans) if (!human.task) {
                    int dist = calc_dist(human.pos, pet.pos);
                    if (chmin(min_dist, dist)) {
                        assigned_human = &human;
                    }
                }
                if (assigned_human) {
                    pet.task->assignee = assigned_human;
                    assigned_human->task = pet.task;
                }
            }
        }

        vector<coord> enum_kill_zone() {
            static int dist[NN];
            
            {
                Fill(dist, inf);
                std::queue<coord> qu;
                qu.push(coord(16, 16));
                dist[coord(16, 16).idx] = 0;
                while (!qu.empty()) {
                    auto u = qu.front(); qu.pop();
                    for (int d = 0; d < 4; d++) {
                        auto v = u.moved(d);
                        if (is_blocked[v.idx] || dist[v.idx] != inf) continue;
                        dist[v.idx] = dist[v.idx] + 1;
                        qu.emplace(v);
                    }
                }
            }

            vector<coord> coords({ 
                {30,1},{1,30},
                {30,3},{3,30},{27,1},{1,27},
                {30,6},{6,30},{24,1},{1,24},
                {30,9},{9,30},{21,1},{1,21},
                {30,12},{12,30},{18,1},{1,18},
                {30,15},{15,30},{15,1},{1,15},
                {30,18},{18,30},{12,1},{1,12},
                {30,21},{21,30},{9,1},{1,9}
                });

            vector<coord> cands;
            for (auto c : coords) if (dist[c.idx] != inf) {
                cands.push_back(c);
            }

            return cands;
        }

        void solve() {

            show();

            assign_tasks();
            update_queue();
            while (turn < MAX_TURN) {
                auto actions = resolve_actions();
                cout << actions << endl;
                load_pet_moves();
                update_queue();
                assign_tasks();
                turn++;
                show();
            }

            dump(enum_kill_zone());

            for (const auto& pet : pets) if (!pet.is_captured) {
                dump(pet);
            }

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

        void show(int delay = 0) const {

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
                if (human.task && human.task->type == Task::Type::CAP) {
                    auto ctask = reinterpret_cast<CapTask*>(human.task);
                    auto ppos = ctask->target->pos;
                    cv::arrowedLine(img, get_point(human.pos), get_point(ppos), cv::Scalar(0, 0, 0));
                }
            }
            cv::imshow("img", img);
            cv::waitKey(delay);
        }
#else
        void show(int delay = 0) const {}
#endif

    };

    void sandbox() {

        TaskGenerator gen;
        auto tasks = gen.generate_sequential_tasks();
        
    }

}

int main() {

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    c2d['R'] = c2d['r'] = 0;
    c2d['U'] = c2d['u'] = 1;
    c2d['L'] = c2d['l'] = 2;
    c2d['D'] = c2d['d'] = 3;

    NSolver::State state(cin, cout);
    state.solve();

    return 0;
}