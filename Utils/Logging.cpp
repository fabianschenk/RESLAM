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
 */
//#undef LOG
#include "Logging.h"

#ifndef NVCC
 #include <iomanip>
#endif

#define I3D_LOGGING_SEPARATOR_A         "|"
#define I3D_LOGGING_SEPARATOR_B         ": "
#define I3D_LOGGING_SEPARATOR_TIME      I3D_LOGGING_SEPARATOR_A

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__)
 #define I3D_LOGGING_PATH_SEPARATOR     '\\'
#else
 #define I3D_LOGGING_PATH_SEPARATOR     '/'
#endif

// Abbreviations of level names
#define I3D_LOGGING_LEVELNAME_DETAIL    "DTL"
#define I3D_LOGGING_LEVELNAME_TRACE     "TRC"
#define I3D_LOGGING_LEVELNAME_DEBUG     "DBG"
#define I3D_LOGGING_LEVELNAME_INFO      "INF"
#define I3D_LOGGING_LEVELNAME_WARNING   "WNG"
#define I3D_LOGGING_LEVELNAME_ERROR     "ERR"
#define I3D_LOGGING_LEVELNAME_FATAL     "FTL"

#if defined(I3D_LOGGING_ENABLE_LEVELNAMES) && !defined(I3D_LOGGING_ENABLE_FILEINFO)
 #undef  I3D_LOGGING_SEPARATOR_A
 #define I3D_LOGGING_SEPARATOR_A        I3D_LOGGING_SEPARATOR_B
#endif

namespace i3d {

LogLevel Logging::threshold_ = trace;

#ifndef WINDOWS
bool Logging::enable_colors_ = true;
#else
bool Logging::enable_colors_ = false;
#endif
bool Logging::enable_timestamp_ = false;
bool Logging::enable_severity_ = true;
bool Logging::enable_filename_ = true;
bool Logging::enable_functionname_ = true;

const char* const Logging::log_colors_[] = { "", I3D_LOGGING_COLOR_FATAL,
    I3D_LOGGING_COLOR_ERROR, I3D_LOGGING_COLOR_WARNING, I3D_LOGGING_COLOR_INFO,
    I3D_LOGGING_COLOR_DEBUG, I3D_LOGGING_COLOR_TRACE, I3D_LOGGING_COLOR_DETAIL };

const char* const Logging::level_indicators_[] = { "",
    I3D_LOGGING_LEVELNAME_FATAL I3D_LOGGING_SEPARATOR_A,
    I3D_LOGGING_LEVELNAME_ERROR I3D_LOGGING_SEPARATOR_A,
    I3D_LOGGING_LEVELNAME_WARNING I3D_LOGGING_SEPARATOR_A,
    I3D_LOGGING_LEVELNAME_INFO I3D_LOGGING_SEPARATOR_A,
    I3D_LOGGING_LEVELNAME_DEBUG I3D_LOGGING_SEPARATOR_A,
    I3D_LOGGING_LEVELNAME_TRACE I3D_LOGGING_SEPARATOR_A,
    I3D_LOGGING_LEVELNAME_DETAIL I3D_LOGGING_SEPARATOR_A };

Logging::Logging(LogLevel level, const std::string &file, int line, const std::string &function, bool append_crlf) :
    append_crlf_(append_crlf)
{
    // Note: The code using log_colors_ and level_indicators_ is efficient but unsafe
    // if level is "abused" or changes in LogLevel are not considered here! Maybe
    // add an assertion to increase code safety.

    // Set logging color if required
    if( enable_colors_ )
        stream_ << log_colors_[level];

    // Add time if required
    if( enable_timestamp_ )
    {
        time_t time_in_sec;
        std::time(&time_in_sec);
        struct tm *local_time = std::localtime(&time_in_sec);

        char fill_char = stream_.fill('0');

#ifdef NVCC
        std::streamsize min_width = stream_.width(2);

        stream_ << local_time->tm_hour << ":"
                << local_time->tm_min << ":"
                << local_time->tm_sec << I3D_LOGGING_SEPARATOR_TIME;

        stream_.width(min_width);
#else
        stream_ << std::setw(2) << local_time->tm_hour << ":"
                << std::setw(2) << local_time->tm_min << ":"
                << std::setw(2) << local_time->tm_sec << I3D_LOGGING_SEPARATOR_TIME;
#endif

        stream_.fill(fill_char);
    }

    // Add level indicator if required
    if( enable_severity_ )
        stream_ << level_indicators_[level];

    // Add file name and line number if required
    if( enable_filename_ && !enable_functionname_ )
    {
        size_t pos = file.find_last_of(I3D_LOGGING_PATH_SEPARATOR);
        stream_ << file.substr(pos != std::string::npos ? pos + 1 : 0)
                << "[" << line << "]" I3D_LOGGING_SEPARATOR_B;
    }
    if( enable_functionname_ )
    {
        size_t pos = file.find_last_of(I3D_LOGGING_PATH_SEPARATOR);
        stream_ << file.substr(pos != std::string::npos ? pos + 1 : 0)
                << "[" << function << "," << line << "]" I3D_LOGGING_SEPARATOR_B;
    }
}

Logging::Logging(LogLevel level) : append_crlf_(false)
{
    // Set logging color if required
    if( enable_colors_ )
        stream_ << log_colors_[level];
}

Logging::~Logging()
{
    if( enable_colors_ )
        stream_ << I3D_LOGGING_COLOR_ALLOFF;

    if( append_crlf_ )
        stream_ << std::endl;

    std::cout << stream_.str();
    std::cout.flush();
}

} // End of i3d namespace
