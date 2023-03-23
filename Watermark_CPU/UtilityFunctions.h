#pragma once

#include <chrono>
namespace timer {
	static std::chrono::time_point<std::chrono::steady_clock> start_timex, cur_timex;
	void start();
	void end();
	double secs_passed();
}