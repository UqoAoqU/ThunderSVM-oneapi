#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <cstdio>
#include <time.h>
#include <string>

class timer {
public:
    long long t, t1;
    bool flag;
    std::string Name;
    timer(const char *);
    ~timer();
    void start();
    void end();
    std::string& get_name() {
        return Name;
    }
    long long get_total_time() {
        return t;
    }
};
#endif