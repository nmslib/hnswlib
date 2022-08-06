#ifndef GRAFT_DIRECTORYSTREAM_H
#define GRAFT_DIRECTORYSTREAM_H

#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "blockbuf.h"

// This class provides a concrete implementation of bufstream for debugging purposes. It uses
// directories as a block device and creates a new file for every block.

namespace graft {

class directorybuf : public blockbuf {
  public:
    // Constructors:
    explicit directorybuf(const std::string& path);
    ~directorybuf() override;

  private:
		static const size_t BLOCK_SIZE = 4;

		size_t device_end() override;
		void device_clear() override;
		size_t block_capacity() override;
		size_t block_size(size_t block_id) override;

		int read(size_t block_id, char_type* buffer, size_t offset) override;
		int write(size_t block_id, char_type* buffer, size_t n) override; 

		// Return a path to a file in the root directory which corresponds to this block.
		std::string get_file(size_t block_id) const;

		// This path to the directory to use as a backing store.
		std::string path_;
};

class idirectorystream : public std::istream {
  public:
    idirectorystream(const std::string& path);
    ~idirectorystream() override;

  private:
    directorybuf buf_;
};

class odirectorystream : public std::ostream {
  public:
    odirectorystream(const std::string& path);
    ~odirectorystream() override;

  private:
    directorybuf buf_;
};

class directorystream : public std::iostream {
  public:
    directorystream(const std::string& path);
    ~directorystream() override;

  private:
    directorybuf buf_;
};

inline directorybuf::directorybuf(const std::string& path) : 
		blockbuf(new char_type[BLOCK_SIZE], new char_type[BLOCK_SIZE]), path_(path) {
	// Create root directory if it doesn't already exist
	mkdir(path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
}

inline directorybuf::~directorybuf() {
	delete[] get_area_;
	delete[] put_area_;
}

inline size_t directorybuf::device_end() {
	// Dictionary file is stored in block zero.
	std::ifstream ifs(get_file(0));
	// Return 1 if dictionary file doesn't exist.
	if (!ifs.is_open()) {
		return 1;
	}
	// Otherwise return the value stored in the file.
	size_t res = 0;
	ifs >> res;
	return res;
}

inline void directorybuf::device_clear() {
	// Write 1 back to block zero.
	std::ofstream ofs(get_file(0));
	ofs << 1;
}

inline size_t directorybuf::block_capacity() {
	return BLOCK_SIZE;
}

inline size_t directorybuf::block_size(size_t block_id) {
	return read(block_id, nullptr, block_capacity()+1);
}

inline int directorybuf::read(size_t block_id, char_type* buffer, size_t offset) {
	// Open backing file for this block (id+1).
	std::ifstream ifs(get_file(block_id+1), std::ios::binary);
	// Reading from a block which was never written is undefined, return 0.
	if (!ifs.is_open()) {
		return 0;
	}
	// Read block size
	size_t block_size = 0;
	ifs.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
	// Only try reading if we asked for an offset within the block_size
	if (offset < block_size) {
		ifs.seekg(offset, std::ios::cur);
		ifs.read(buffer+offset, block_size-offset);
	}
	return block_size;
}

inline int directorybuf::write(size_t block_id, char_type* buffer, size_t n) {
	// Open backing file for this block (id+1).
	const auto file = get_file(block_id+1);
	std::ofstream ofs(file, std::ios::binary);
	// Write blocksize and new data.
	ofs.write(reinterpret_cast<char*>(&n), sizeof(n));
	ofs.write(buffer, n);
	// Update device end if necessary
	if (block_id >= device_end()) {
		std::ofstream ofs(get_file(0));
		ofs << (block_id+1);
	}
	// Return the size of this block
	return n;
}

std::string directorybuf::get_file(size_t block_id) const {
	std::ostringstream oss;
	oss << path_ << "/" << block_id;
	return oss.str();
}

inline idirectorystream::idirectorystream(const std::string& path) : std::istream(&buf_), buf_(path) {
	buf_.open(std::ios_base::in | std::ios_base::app);
}

inline idirectorystream::~idirectorystream() {
	buf_.close();
}

inline odirectorystream::odirectorystream(const std::string& path) : std::ostream(&buf_), buf_(path) {
	buf_.open(std::ios_base::out | std::ios_base::trunc);
}

inline odirectorystream::~odirectorystream() {
	buf_.close();
}

inline directorystream::directorystream(const std::string& path) : std::iostream(&buf_), buf_(path) {
	buf_.open(std::ios_base::in | std::ios_base::out | std::ios_base::app);
}

inline directorystream::~directorystream() {
	buf_.close();
}

} // namespace graft

#endif
