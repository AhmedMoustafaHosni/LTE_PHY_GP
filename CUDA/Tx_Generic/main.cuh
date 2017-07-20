#include <chrono>

#define timerInit(); std::chrono::steady_clock::time_point start; std::chrono::steady_clock::time_point end;
#define startTimer(); start = std::chrono::steady_clock::now();
#define stopTimer(msg); end = std::chrono::steady_clock::now(); printf(msg, std::chrono::duration_cast<std::chrono::nanoseconds> (end - start).count()/1000000.0) ;

//Example for timer macros usage:
//	timerInit();
//	startTimer();
//  ...do_something();
//  stopTimer("Time= %.10f ms\n", elapsed);
//  ...at the very end
//  destroyTimers();