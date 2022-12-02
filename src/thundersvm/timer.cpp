#include "thundersvm/timer.h"

using std::cout, std::endl;
timer::timer(const char *s):Name(s), t(0), flag(0){
}
timer::~timer() {
    cout << "the total time of " << get_name() << ":" << (get_total_time()) << '\n'; 
}
void timer::start() {
    flag = 1;
    t1 = clock();
}
void timer::end() {
    if (flag == 1)
        t += clock() - t1;
    // else 
    //     cout << "end() doesn't match a start()\n";
    flag = 0;
}
