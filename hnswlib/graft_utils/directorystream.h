#ifndef GRAFT_DIRECTORYSTREAM_H
#define GRAFT_DIRECTORYSTREAM_H

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "blockbuf.h"

// This class provides a concrete implementation of bufstream for debugging purposes. It uses
// directories as a block device and creates a new file for every block.

namespace graft {

// TODO(eschkufz): This class could really use an is_open() method. Currently, we'll fail 
//	less than gracefully if we attempt to write to a directory that doesn't exist.

class directorybuf : public blockbuf {
  public:
    // Constructors:
    explicit directorybuf(const std::string& path, std::ios_base::openmode mode);
    ~directorybuf() override;

  private:
		static const size_t BLOCK_SIZE = 1024;

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
    ~idirectorystream() override = default;

  private:
    directorybuf buf_;
};

class odirectorystream : public std::ostream {
  public:
    odirectorystream(const std::string& path);
    ~odirectorystream() override = default;

  private:
    directorybuf buf_;
};

class directorystream : public std::iostream {
  public:
    directorystream(const std::string& path);
    ~directorystream() override = default;

  private:
    directorybuf buf_;
};

inline directorybuf::directorybuf(const std::string& path, std::ios_base::openmode mode) : 
		blockbuf(new char_type[BLOCK_SIZE], new char_type[BLOCK_SIZE], BLOCK_SIZE), 
		path_(path) {
	// Delete root directory in truncate mode.
	if (mode & std::ios_base::trunc) {
		std::ostringstream oss;
		oss << "rm -rf " << path;
		std::system(oss.str().c_str());		
	} 
	// Create root directory in output mode.
	if (mode & std::ios_base::out) {
		std::ostringstream oss;
		oss << "mkdir " << path;
		std::system(oss.str().c_str());
	}
}

inline directorybuf::~directorybuf() {
	delete[] get_area_;
	delete[] put_area_;
}

inline int directorybuf::read(size_t block_id, char_type* buffer, size_t offset) {
	// Open backing file for this block.
	const auto file = get_file(block_id);
	std::ifstream ifs(file, std::ios::binary);
	// If the file didn't open, the block doesn't exist and the size is zero.
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
	// Open backing file for this block.
	const auto file = get_file(block_id);
	std::ofstream ofs(file, std::ios::binary);
	// Write blocksize and new data.
	ofs.write(reinterpret_cast<char*>(&n), sizeof(n));
	ofs.write(buffer, n);
	return n;
}

std::string directorybuf::get_file(size_t block_id) const {
	std::ostringstream oss;
	oss << path_ << "/" << block_id;
	return oss.str();
}

inline idirectorystream::idirectorystream(const std::string& path) : 
	std::istream(&buf_), 
	buf_(path, std::ios_base::in | std::ios_base::app) { 
}

inline odirectorystream::odirectorystream(const std::string& path) : 
	std::ostream(&buf_), 
	buf_(path, std::ios_base::out | std::ios_base::trunc) { 
}

inline directorystream::directorystream(const std::string& path) : 
	std::iostream(&buf_), 
	buf_(path, std::ios_base::in | std::ios_base::out | std::ios_base::app) { 
}

} // namespace graft

#endif
