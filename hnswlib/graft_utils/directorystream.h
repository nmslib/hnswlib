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

class directorybuf : public blockbuf {
  public:
    // Constructors:
    explicit directorybuf(const std::string& path);
    ~directorybuf() override = default;

  private:
		int read(size_t block_id, char_type* buffer, size_t offset) override;
		int write(size_t block_id, char_type* buffer, size_t n) override; 

		std::string get_file(size_t block_id) const;

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

inline directorybuf::directorybuf(const std::string& path) : blockbuf(1024), path_(path) { 
	std::ostringstream oss;
	oss << "mkdir " << path;
	std::system(oss.str().c_str());
}

int directorybuf::read(size_t block_id, char_type* buffer, size_t offset) {
	// Open backing file for this block.
	const auto file = get_file(block_id);
	std::ifstream ifs(file, std::ios::binary);
	// If the file didn't open, the block doesn't exist and the size is zero.
	if (!ifs.is_open()) {
		return 0;
	}
	// Read block size
	size_t blocksize = 0;
	ifs.read(reinterpret_cast<char*>(&blocksize), sizeof(blocksize));
	// Only try reading if we asked for an offset within the blocksize
	if (offset < blocksize) {
		ifs.seekg(offset, std::ios::cur);
		ifs.read(buffer+offset, blocksize-offset);
	}
	return blocksize;
}

int directorybuf::write(size_t block_id, char_type* buffer, size_t n) {
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

inline idirectorystream::idirectorystream(const std::string& path) : std::istream(&buf_), buf_(path) { }

inline odirectorystream::odirectorystream(const std::string& path) : std::ostream(&buf_), buf_(path) { }

inline directorystream::directorystream(const std::string& path) : std::iostream(&buf_), buf_(path) { }

} // namespace graft

#endif
