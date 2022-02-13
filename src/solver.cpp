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
#define ENABLE_DUMP
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
constexpr char BLOCK = '#';
constexpr char EMPTY = '.';
constexpr char HUMAN = 'H';



namespace NSolver {

    // pet P 匹とする
    // 何もしない場合のスコアは s0 = 2^(-P)
    // 領域を 4 分割した場合の平均スコアは s1 = 0.25 * 2^(-P/4) = 2^(-P/4 - 2)
    // s1 / s0 = 2^(P-P/4 - 2)
    // P=10 なら 2^5.5 ~ 45 倍
    // P=20 なら 2^13 ~ 8192 倍のスコア増加が見込める

    struct Pet {
        int id, y, x, t;
        Pet(int id = -1, int y = -1, int x = -1, int t = -1) : id(id), y(y), x(x), t(t) {}
    };

    struct Human {
        int id, y, x;
        Human(int id = -1, int y = -1, int x = -1) : id(id), y(y), x(x) {}
    };

    struct Point {
        int y, x;
        Point(int y = -1, int x = -1) : y(y), x(x) {}
        string stringify() const {
            return format("(%d,%d)", y, x);
        }
    };

    struct Cmd {
        enum struct Type {
            MOVE, BLOCK, WAIT
        };
        Type type;
        int y, x, mask, w;
        Cmd(Type type, int y, int x, int mask, int wait) : type(type), y(y), x(x), mask(mask), w(wait) {}
        static Cmd move(int y, int x) {
            return Cmd(Type::MOVE, y, x, -1, -1);
        }
        static Cmd block(int mask) {
            return Cmd(Type::BLOCK, -1, -1, mask, -1);
        }
        static Cmd wait(int w) {
            return Cmd(Type::WAIT, -1, -1, -1, w);
        }
        string stringify() const {
            switch (type)
            {
            case Cmd::Type::MOVE:
                return format("move(%d,%d)", y, x);
            case Cmd::Type::BLOCK:
                return format("block(%s)", std::bitset<4>(mask).to_string().c_str());
            case Cmd::Type::WAIT:
                return format("wait(%d)", w);
            default:
                return "";
            }
            return "";
        }
    };

    struct State {

        static constexpr bool debug = false;

        std::istream& in;
        std::ostream& out;

        vector<Pet> pets;
        vector<Human> humans;

        vector<std::deque<Cmd>> cmd_queue;


        bool blocked[N][N];
        int human_count[N][N];
        int pet_count[N][N];

        bool blocked_tmp[N][N];
        int human_count_tmp[N][N];

        State(std::istream& in, std::ostream& out) : in(in), out(out) { init(); }

        void init() {
            if (debug) cerr << format("--- %s called ---\n", __FUNCTION__);
            Fill(blocked, false);
            Fill(human_count, 0);
            Fill(pet_count, 0);
            int num_pets; in >> num_pets;
            if (debug) cerr << format("num_pets: %d\n", num_pets);
            for (int pid = 0; pid < num_pets; pid++) {
                int y, x, t;
                cin >> y >> x >> t;
                x--; y--; t--;
                if (debug) cerr << format("position of pet %d: (%d, %d)\n", pid, y, x);
                pets.emplace_back(pid, y, x, t);
                pet_count[y][x]++;
            }
            int num_humans; in >> num_humans;
            if (debug) cerr << format("num_humans: %d\n", num_humans);
            for (int hid = 0; hid < num_humans; hid++) {
                int y, x;
                cin >> y >> x;
                x--; y--;
                if (debug) cerr << format("position of human %d: (%d, %d)\n", hid, y, x);
                humans.emplace_back(hid, y, x);
                human_count[y][x]++;
            }
            if (debug) cerr << format("--- %s end ---\n", __FUNCTION__);
        }

        vector<string> load() {
            vector<string> pet_moves(pets.size());
            cin >> pet_moves;
            for (int pid = 0; pid < pets.size(); pid++) {
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
            if (!is_inside(y, x) || blocked_tmp[y][x] || human_count_tmp[y][x] || pet_count[y][x]) return false;
            for (int d = 0; d < 4; d++) {
                int ny = y + dy[d], nx = x + dx[d];
                if (!is_inside(ny, nx)) continue;
                if (pet_count[ny][nx]) return false;
            }
            return true;
        }

        char calc_move(int hid, int gy, int gx) {
            static constexpr int inf = INT_MAX / 8;
            // 目的地からの距離を計算
            static int dist[N][N];
            if (humans[hid].y == gy && humans[hid].x == gx) return '.';
            Fill(dist, inf);
            std::queue<Point> qu;
            qu.emplace(gy, gx);
            dist[gy][gx] = 0;
            while (!qu.empty()) {
                auto [y, x] = qu.front(); qu.pop();
                for (int d = 0; d < 4; d++) {
                    int ny = y + dy[d], nx = x + dx[d];
                    if (!is_inside(ny, nx) || blocked_tmp[ny][nx] || pet_count[ny][nx] || dist[ny][nx] != inf) continue;
                    dist[ny][nx] = dist[y][x] + 1;
                    qu.emplace(ny, nx);
                }
            }
            auto [_, sy, sx] = humans[hid];
            if (dist[sy][sx] == inf) return '.'; // 移動不可
            
            int min_dist = inf;
            vector<int> cands;
            for (int d = 0; d < 4; d++) {
                int ny = sy + dy[d], nx = sx + dx[d];
                if (!is_inside(ny, nx) || blocked_tmp[ny][nx] || pet_count[ny][nx] || dist[ny][nx] > min_dist) continue;
                if (dist[ny][nx] < min_dist) {
                    min_dist = dist[ny][nx];
                    cands.clear();
                }
                cands.push_back(d);
            }
            int d = cands[rnd.next_int(cands.size())];
            human_count_tmp[sy + dy[d]][sx + dx[d]]++;
            return d2C[d];
        }

        char calc_block(int hid, int mask) {
            auto [_, y, x] = humans[hid];
            for (int d = 0; d < 4; d++) if (mask >> d & 1) {
                if (can_place(y + dy[d], x + dx[d])) {
                    blocked_tmp[y + dy[d]][x + dx[d]] = true;
                    return d2c[d];
                }
            }
            return '.';
        }

        char calc_move(int hid) {
            auto& cqu = cmd_queue[hid];
            if (cqu.empty()) return '.';
            auto cmd = cqu.front();
            auto type = cmd.type;
            switch (type)
            {
            case Cmd::Type::MOVE:
                return calc_move(hid, cmd.y, cmd.x);
            case Cmd::Type::BLOCK:
                return calc_block(hid, cmd.mask);
            case Cmd::Type::WAIT:
                return '.';
            }
            return '.';
        }

        string calc_moves() {
            memcpy(blocked_tmp, blocked, sizeof(bool) * N * N);
            memcpy(human_count_tmp, human_count, sizeof(int) * N * N);
            string moves(humans.size(), '.');
            for (int hid = 0; hid < humans.size(); hid++) {
                moves[hid] = calc_move(hid);
            }
            return moves;
        }

        void do_moves(const string& moves) {
            for (int hid = 0; hid < humans.size(); hid++) {
                auto& [_, y, x] = humans[hid];
                char c = moves[hid];
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
                auto& cqu = cmd_queue[hid];
                if (cqu.empty()) continue;
                auto [_, y, x] = humans[hid];
                auto& cmd = cqu.front();
                auto type = cmd.type;
                switch (type)
                {
                case Cmd::Type::MOVE:
                {
                    if (x == cmd.x && y == cmd.y) {
                        cqu.pop_front();
                    }
                    break;
                }
                case Cmd::Type::BLOCK:
                {
                    int mask = cmd.mask;
                    bool completed = true;
                    for (int d = 0; d < 4; d++) if (mask >> d & 1) {
                        int ny = y + dy[d], nx = x + dx[d];
                        if (!is_inside(ny, nx)) continue;
                        if (!blocked[ny][nx]) {
                            completed = false;
                            break;
                        }
                    }
                    if (completed) cqu.pop_front();
                    break;
                }
                case Cmd::Type::WAIT:
                {
                    cmd.w--;
                    if (!cmd.w) {
                        cqu.pop_front();
                    }
                }
                }
            }
        }

        void solve() {

            cmd_queue.resize(humans.size());

            vector<Point> corner({ {0, 0}, {N - 1, 0}, {N - 1, N - 1}, {0, N - 1} });
            for (int hid = 4; hid < humans.size(); hid++) {
                auto [y, x] = corner[hid % 4];
                cmd_queue[hid].push_back(Cmd::move(y, x));
            }
            cmd_queue[0].push_back(Cmd::move(0, 14));//r
            cmd_queue[0].push_back(Cmd::wait(60));
            for (int y = 0; y <= 14; y++) {
                cmd_queue[0].push_back(Cmd::move(y, 14));
                cmd_queue[0].push_back(Cmd::block(0b0001));
            }
            cmd_queue[1].push_back(Cmd::move(14, N - 1));//d
            cmd_queue[1].push_back(Cmd::wait(60));
            for (int x = N - 1; x >= 16; x--) {
                cmd_queue[1].push_back(Cmd::move(14, x));
                cmd_queue[1].push_back(Cmd::block(0b1000));
            }
            cmd_queue[2].push_back(Cmd::move(N - 1, 15));//l
            cmd_queue[2].push_back(Cmd::wait(60));
            for (int y = N - 1; y >= 15; y--) {
                cmd_queue[2].push_back(Cmd::move(y, 15));
                cmd_queue[2].push_back(Cmd::block(0b0100));
            }
            cmd_queue[3].push_back(Cmd::move(15, 0));//u
            cmd_queue[3].push_back(Cmd::wait(60));
            for (int x = 0; x < 14; x++) {
                cmd_queue[3].push_back(Cmd::move(15, x));
                cmd_queue[3].push_back(Cmd::block(0b0010));
            }

            for (int turn = 0; turn < MAX_TURN; turn++) {
                if (debug) {
                    cerr << format("--- turn %d ---\n", turn);
                }
                auto moves = calc_moves();
                dump(turn, moves);
                if (debug) cerr << format("move %3d: %s\n", turn, moves.c_str());
                do_moves(moves);
                cout << moves << endl;
                load();
                update_queue();
            }
        }

    };

}

#ifdef HAVE_OPENCV_HIGHGUI

namespace NManual {

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

    c2d['R'] = c2d['r'] = 0;
    c2d['U'] = c2d['u'] = 1;
    c2d['L'] = c2d['l'] = 2;
    c2d['D'] = c2d['d'] = 3;

    //NManual::State state(cin, cout);
    //state.play();

    NSolver::State state(cin, cout);

    state.solve();

    return 0;
}