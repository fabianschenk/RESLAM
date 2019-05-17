/*
 * logging.h
 *
 *  Created on: Aug 12, 2011
 *      Author: Matthias Hotz
 *
 * The implementation below is tailored to the needs of the I3D library.
 * In particular the logging information is written to the standard output
 * and the run-time overhead is kept at a minimum. Furthermore, the logging
 * information is only written to the output if its severity level is above
 * some specific threshold.
 *
 * Usage:
 * ------
 *
 *  -> A message may be logged using the macro I3D_LOG(.). As a parameter the
 *     macro takes the severity level of the message being one of those
 *     specified in the enumeration LogLevel (see code below). The "return
 *     value" of this macro is a stream object and may be used as such:
 *
 *       I3D_LOG(warning) << "Warning message";
 *
 *     If several values should be logged to the same log line (e.g. as
 *     sometimes the case with loops), the macros LOG_PREFIX(.) (works
 *     like I3D_LOG(.) but does not add a CR/LF) and LOG_BARE(.) (logs only
 *     the text without a prefix and CR/LF).
 *
 *     ATTENTION: If the severity of the message is below the threshold
 *     the code to the right of the macro is NOT EXECUTED. Thus, any
 *     operations necessary for the "logic" of the program may not be
 *     placed in a log message line.
 *
 *  -> The logging threshold is set using the macro LOG_THRESHOLD(.):
 *
 *       LOG_THRESHOLD(info);
 *
 *     Furthermore, a compile-time threshold may be specified via
 *     I3D_LOGGING_STATIC_LEVEL_LIMIT defined below (Change the according
 *     define-directive below). Any log messages with a severity below
 *     this threshold are removed during compile-time for increased
 *     performance. This may also be used to completely disable the
 *     logging system by setting I3D_LOGGING_STATIC_LEVEL_LIMIT to 0.
 *
 *  -> The format of the logging output is defined using the macro
 *     LOG_FORMAT(.), where the individual prefix elements may be
 *     activated or deactivated by passing a corresponding boolean
 *     value true or false, respectively.
 *
 *       LOG_FORMAT(colors, timestamp, severity, filename);
 *
 *
 * Background information:
 * -----------------------
 *
 * Before writing this logging system I glanced over some other logging
 * systems like Boost.Log, Pantheios, log4cpp, log4cplus, C++ Trivial Log
 * and EZLogger. Most of them bring a quite large amount of overhead with
 * them (especially the first four), but their flexibility is not required
 * in the I3D library. Due to that this extremely lightweight logging system
 * was written, which is based on the ideas of Petru Marginean presented in
 *   http://drdobbs.com/cpp/201804215
 *   http://drdobbs.com/cpp/221900468 .
 * Three ideas therefrom were incorporated here:
 *   1) A macro with an if-else-construct is used for "thresholding"
 *   2) Thread safety is achieved by created by creating a temporary
 *      object, cumulating the output and writing the complete log message
 *      in one atomic operation to the output stream during object
 *      destruction (lifetime limited to else-branch).
 *   3) A compile-time threshold is used to increase performance and
 *      enable the deactivation of the logging system.
 * Furthermore, the style of the logging macro was chosen analogous to
 * Boost.Log's BOOST_LOG_TRIVIAL, enabling a change to Boost.Log by
 * redefining the macros below if some later time its capability is
 * really required.
 */

#ifndef I3D_CORE_LOGGING_H_
#define I3D_CORE_LOGGING_H_

#include <sstream>
#include <string>
#include <ostream>
#include <iostream>
#include <ctime>
#include <chrono>

// Define the compile-time log threshold
#define I3D_LOGGING_STATIC_LEVEL_LIMIT ::i3d::detail

// Color console output using escape codes:
// (http://en.wikipedia.org/wiki/ANSI_escape_code)
//
// 0m - all attibutes off
// 1m - intensity
// 4m - underscore
// 5m - blinking
// 7m - reverse
//
// 30m - black
// 31m - red
// 32m - green
// 33m - yellow
// 34m - blue
// 35m - pink
// 36m - cyan
// 37m - white
//
#define I3D_LOGGING_COLOR_FATAL         "\33[35m"
#define I3D_LOGGING_COLOR_ERROR         "\33[31m"
#define I3D_LOGGING_COLOR_WARNING       "\33[33m"
#define I3D_LOGGING_COLOR_INFO          "\33[32m"
#define I3D_LOGGING_COLOR_DEBUG         "\33[36m"
#define I3D_LOGGING_COLOR_TRACE         ""
#define I3D_LOGGING_COLOR_DETAIL        "\33[34m"
#define I3D_LOGGING_COLOR_ALLOFF        "\33[0m"

namespace i3d {

enum LogLevel {
    show_all = 8,
    detail = 7,
    trace = 6,
    debug = 5,
    info = 4,
    warning = 3,
    error = 2,
    fatal = 1,
    nothing = 0
};

class Logging {

public:
    Logging(LogLevel level, const std::string &file, int line, const std::string &function, bool append_crlf);
    Logging(LogLevel level);
    ~Logging();
    std::ostream& getStream() { return stream_; }
    std::ostream& addPrefix(const std::string &file, int line);

    static void setLogThreshold(LogLevel level) { threshold_ = level; }
    static LogLevel getLogThreshold() { return threshold_; }

    static void displayColor(bool enable) { enable_colors_ = enable; }
    static void displayTimeStamp(bool enable) { enable_timestamp_ = enable; }
    static void displaySeverity(bool enable) { enable_severity_ = enable; }
    static void displayFileName(bool enable) { enable_filename_ = enable; }

private:
    std::ostringstream stream_;
    bool append_crlf_;
    static LogLevel threshold_;
    static bool enable_colors_;
    static bool enable_timestamp_;
    static bool enable_severity_;
    static bool enable_filename_;
    static bool enable_functionname_;
    static const char* const log_colors_[];
    static const char* const level_indicators_[];

    Logging();
    Logging(const Logging&);
    Logging& operator=(const Logging&);
};


} // End of i3d namespace

#define LOG_PREFIX(level)                                \
    if( level > I3D_LOGGING_STATIC_LEVEL_LIMIT ) ;       \
    else if( level > i3d::Logging::getLogThreshold() ) ; \
    else i3d::Logging(level, __FILE__, __LINE__, __FUNCTION__, false).getStream()

#define LOG_BARE(level)                                  \
    if( level > I3D_LOGGING_STATIC_LEVEL_LIMIT ) ;       \
    else if( level > i3d::Logging::getLogThreshold() ) ; \
    else i3d::Logging(level).getStream()

#define I3D_LOG(level)									 \
	if( level > I3D_LOGGING_STATIC_LEVEL_LIMIT ) ;       \
	else if( level > i3d::Logging::getLogThreshold() ) ; \
    else i3d::Logging(level, __FILE__, __LINE__, __FUNCTION__, true).getStream()


inline void LOG_THRESHOLD(::i3d::LogLevel level)
{
    ::i3d::Logging::setLogThreshold(level);
}

inline void LOG_FORMAT(bool colors, bool timestamp, bool severity, bool filename)
{
#if WIN32
    ::i3d::Logging::displayColor(false);
#else
	::i3d::Logging::displayColor(colors);
#endif
    ::i3d::Logging::displayTimeStamp(timestamp);
    ::i3d::Logging::displaySeverity(severity);
    ::i3d::Logging::displayFileName(filename);
}


#endif
