#pragma once
#include <windows.h>
#include <chrono>
#include <iostream>
struct HighResClock
{
    typedef long long rep;
    typedef std::nano period;
    typedef std::chrono::duration<rep, period> duration;
    typedef std::chrono::time_point<HighResClock> time_point;
    static const bool is_steady = true;

    static time_point now();
};

class Timer
{
public:
    enum DURATION : uint8_t
    {
        SECONDS = 0,
        MILLISECONDS,
        MICROSECONDS,
        NANOSECONDS
    };
    Timer(bool en, DURATION printT);
    void printElapsed(const char *tag);

private:
    std::chrono::time_point<HighResClock> start;
    bool enabled = true;
    DURATION printType;
};