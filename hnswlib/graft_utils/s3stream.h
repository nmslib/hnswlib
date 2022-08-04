#ifndef GRAFT_S3STREAM_H
#define GRAFT_S3STREAM_H

#include "blockstream.h"

// This class provides a c++ stream interface to s3 objects. The main complication
// here is that s3 doesn't support appends. As a result, we treat an s3 directory
// as a block device where a file is spread out across multiple blocks.

namespace graft {

class s3buf : public blockbuf {
  public:
    // Constructors:
    explicit s3buf();
    ~s3buf() override = default;

  private:
		std::streamsize read(size_t block_id, char_type* buffer) override;
		std::streamsize write(size_t block_id, char_type* buffer, const size_t begin, const size_t end) override; 
};

class is3dstream : public std::istream {
  public:
    is3stream();
    ~is3stream() override = default;

  private:
    s3buf buf_;
};

class os3stream : public std::ostream {
  public:
    os3stream();
    ~os3stream() override = default;

  private:
    s3buf buf_;
};

class s3stream : public std::iostream {
  public:
    s3stream();
    ~s3stream() override = default;

  private:
    s3buf buf_;
};

inline s3buf::s3buf() { }

std::streamsize s3buf::read(size_t block_id, char_type* buffer) {
	return 0;
}

std::streamsize s3buf::write(size_t block_id, char_type* buffer, const size_t begin, const size_t end) {
	return 0;
}

inline is3stream::is3stream() : std::istream(&buf_), buf_() { }

inline os3stream::os3stream() : std::ostream(&buf_), buf_() { }

inline s3stream::s3stream() : std::iostream(&buf_), buf_() { }

} // namespace graft

#endif
