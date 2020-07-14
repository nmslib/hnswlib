#pragma once

namespace hnswlib {
#if defined(__unix__) || defined(__APPLE__)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

    class MMap {
    public:
        char* load(const std::string &location) {
            int fd = open(location.c_str(), O_RDONLY);
            if (fd < 0) {
                throw std::runtime_error("Cannot open file");
            }

            length_ = lseek(fd, 0, SEEK_END);
            address_ = (char *) mmap(NULL, length_, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0);
            close(fd);

            if (address_ == (char *) -1) {
                throw std::runtime_error("Unable to mmap file");
            }

            return address_;
        }

        bool isValid() {
            return address_ != (char *) -1;
        }

        bool isInBuffer(char *ptr) {
            return address_ < (ptr + length_);
        }

        ~MMap() {
            if (isValid()) {
                munmap(address_, length_);
            }
        }

    private:
        char *address_ = (char *) -1;
        size_t length_ = 0;
    };
#else
    class MMap {
    public:
        char* load(const std::string &location) {
            throw std::runtime_error("mmap not supported for this OS");
        }

        bool isValid() {
            return false;
        }
    };
#endif
}
