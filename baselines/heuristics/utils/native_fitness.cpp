#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// No fast-math: feasibility and timing follow the Python reference evaluator.
extern "C" int score_routes(
    int n, int count, const int64_t* tasks, const int64_t* mother_pre,
    const int64_t* daughter_pre, const double* joins, const double* pickups,
    const double* legs, const double* due, const double* operation,
    double speed, double attach, double detach, double handling,
    double weight, double* output) {
    const int width = n + 1;
    std::vector<unsigned char> done(width, 0), active(width, 0);
    std::vector<double> mother_end(width, 0.0), daughter_end(width, 0.0);
    done[0] = 1;
    for (int i = 0; i < count; ++i) {
        if (tasks[i] < 1 || tasks[i] > n || active[tasks[i]]) return 0;
        active[tasks[i]] = 1;
    }
    for (int i = 0; i < count; ++i) {
        int t = tasks[i], m = mother_pre[t], d = daughter_pre[t];
        if (m < 0 || m > n || d < 0 || d > n ||
            (m && !active[m]) || (d && !active[d])) return 0;
    }
    double distance = 0.0, tardiness = 0.0;
    int completed = 0;
    while (completed < count) {
        bool progressed = false;
        for (int i = 0; i < count; ++i) {
            const int t = tasks[i], m = mother_pre[t], d = daughter_pre[t];
            if (done[t] || !done[m] || !done[d]) continue;
            double join = joins[m * width + d], pickup = pickups[d * width + t];
            double departure = std::max(mother_end[m] + join / speed, daughter_end[d]);
            mother_end[t] = departure + attach + pickup / speed + handling + legs[t] / speed + detach;
            daughter_end[t] = mother_end[t] + operation[t];
            distance += join + pickup + legs[t];
            tardiness += std::max(0.0, daughter_end[t] - due[t]);
            done[t] = 1;
            ++completed;
            progressed = true;
        }
        if (!progressed) return 0;
    }
    output[0] = weight * distance + (1.0 - weight) * tardiness;
    output[1] = distance;
    output[2] = tardiness;
    return 1;
}
