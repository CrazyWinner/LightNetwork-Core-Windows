#include "HighResClock.h"

const long long g_Frequency = []() -> long long {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    return frequency.QuadPart;
}();

HighResClock::time_point HighResClock::now()
{
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return time_point(duration(count.QuadPart * static_cast<rep>(period::den) / g_Frequency));
}

Timer::Timer(bool en, DURATION printT)
{
    this->enabled = en;
    this->start = HighResClock::now();
    this->printType = printT;
}
void Timer::printElapsed(const char *tag)
{
    if (enabled)
    {
        auto end = HighResClock::now();
        switch (this->printType)
        {
        case SECONDS:
        {
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
            std::cout << tag << ":" << duration.count() << "s" << std::endl;
            break;
        }
        case MILLISECONDS:
        {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << tag << ":" << duration.count() << "ms" << std::endl;
            break;
        }
        case MICROSECONDS:
        {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << tag << ":" << duration.count() << "ns" << std::endl;
            break;
        }
        case NANOSECONDS:
        {
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            std::cout << tag << ":" << duration.count() << "ns" << std::endl;
            break;
        }
        }

        start = HighResClock::now();
    }
}