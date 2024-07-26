#include "UtilityFunctions.hpp"
#include <chrono>

//χρονομέτρηση
namespace timer {
	void start() {
		start_timex = std::chrono::high_resolution_clock::now();
	}
	void end() {
		cur_timex = std::chrono::high_resolution_clock::now();
	}
	double secs_passed() {
		return (double)std::chrono::duration_cast<std::chrono::microseconds>(cur_timex - start_timex).count() / 1000000;
	}
}