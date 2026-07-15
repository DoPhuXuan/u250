#ifndef BERT_STREAM_COMPAT_H
#define BERT_STREAM_COMPAT_H

// Synthesis uses the vendor stream class.  The full-model C-simulation has a
// real feedback cycle and therefore runs the five persistent kernels in host
// threads.  Vitis HLS 2022.1 ships a thread-unsafe hls::stream C model, so the
// test-only macro selects a blocking, thread-safe queue with the same API.
#ifdef BERT_CSIM_THREAD_SAFE_STREAM
#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class bert_csim_stream {
public:
    explicit bert_csim_stream(const char * = 0, std::size_t capacity = 0)
        : capacity_(capacity) {}

    void write(const T &value)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        writable_.wait(lock, [this] {
            return capacity_ == 0 || queue_.size() < capacity_;
        });
        queue_.push(value);
        lock.unlock();
        ready_.notify_one();
    }

    T read()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return !queue_.empty(); });
        T value = queue_.front();
        queue_.pop();
        lock.unlock();
        writable_.notify_one();
        return value;
    }

    bool empty()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable writable_;
    std::size_t capacity_;

    bert_csim_stream(const bert_csim_stream &);
    bert_csim_stream &operator=(const bert_csim_stream &);
};

template <typename T>
using bert_stream_t = bert_csim_stream<T>;

#else
#include <hls_stream.h>

template <typename T>
using bert_stream_t = hls::stream<T>;
#endif

#endif
