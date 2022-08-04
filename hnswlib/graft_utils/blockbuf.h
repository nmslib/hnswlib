#ifndef GRAFT_BLOCKBUF_H
#define GRAFT_BLOCKBUF_H

#include <iostream>
#include <streambuf>
#include <vector>

// This class provides a c++ stream interface to a block storage device. The
// underlying abstraction to that device is that blocks can be read/written
// atomically.

namespace graft {

class blockbuf : public std::streambuf {
  public:
    // Typedefs:
    typedef std::streambuf::char_type char_type;
    typedef std::streambuf::traits_type traits_type;
    typedef std::streambuf::int_type int_type;
    typedef std::streambuf::pos_type pos_type;
    typedef std::streambuf::off_type off_type;
   
    // Constructors:
    explicit blockbuf(size_t blocksize);
    ~blockbuf() override = default;

	protected:
		// Block storage interface
		// Read block into buffer beginning at offset, return the size of the block, or -1 for error.
		virtual int read(size_t block_id, char_type* buffer, size_t offset) = 0;
		// Write buffer to a block, return the size of the block, or -1 for error.
		virtual int write(size_t block_id, char_type* buffer, size_t n) = 0; 

  private:
    // Get/Input/Read Area
    std::vector<char_type> get_area_;
		int get_id_;
    // Put/Output/Write Area
    std::vector<char_type> put_area_;
		int put_id_;
		// Buffer
		// Blocksize
		size_t blocksize_;

    // Locales: 
    void imbue(const std::locale& loc) override;

    // Positioning:
    blockbuf* setbuf(char_type* s, std::streamsize n) override;
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override;
    int sync() override;

    // Get Area:
    std::streamsize showmanyc() override;
    int_type underflow() override;
    int_type uflow() override;
    std::streamsize xsgetn(char_type* s, std::streamsize count) override;

    // Put Area:
    std::streamsize xsputn(const char_type* s, std::streamsize count) override;
    int_type overflow(int_type c = traits_type::eof()) override;

    // Undo:
    int_type pbackfail(int_type c = traits_type::eof()) override;

		// Put Area Helper Methods:
		// Push back the end of the put area to make up to blocksize bytes of n available.
		int extend_put_area(size_t n);
		// Sync the put area, move to the next block id, and make up to blocksize bytes of n available.
		int bump_put_area(size_t n);
};

inline blockbuf::blockbuf(size_t blocksize) : get_area_(blocksize), get_id_(-1), put_area_(blocksize), put_id_(0), blocksize_(blocksize) {
	// Configure get and put areas to fault on first read/write
  setg(get_area_.data(), get_area_.data(), get_area_.data());
  setp(put_area_.data(), put_area_.data()+1);
}

inline void blockbuf::imbue(const std::locale& loc) {
  // Default implementation. Does nothing.
  (void) loc;
}

inline blockbuf* blockbuf::setbuf(char_type* s, std::streamsize n) {
  // Default implementation. Does nothing.
  (void) s;
  (void) n;
  return this;
}

inline blockbuf::pos_type blockbuf::seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) {
  // Does nothing.
	// TODO(eschkufz): Update get/put, page in new blocks if necessary.
  (void) off;
  (void) dir;
  (void) which;
  return pos_type(off_type(-1));
}

inline blockbuf::pos_type blockbuf::seekpos(pos_type pos, std::ios_base::openmode which) {
	// Does nothing.
	// TODO(eschkufz): Update get/put, page in new blocks if necessary.
  (void) pos;
  (void) which;
  return pos_type(off_type(-1));
}

inline int blockbuf::sync() {
	// If the put area is empty, there's nothing to do.
	if (pptr() == pbase()) {
		return 0;
	}
	// If the put area doesn't span a full block, we need to read the part of the block we're missing.
	const size_t offset = pptr()-pbase();
	auto size = offset;
	if (size != blocksize_) {
		size = read(put_id_, put_area_.data(), offset);
		if (size == -1) {
			return -1;
		}
		size = std::max(size, offset);
		setp(pbase(), pbase()+size);
		pbump(offset);
	}
	// Write back 
	if (write(put_id_, put_area_.data(), size) == -1) {
		return -1;
	}
	// If get and put areas refer to the same block, copy the put area to the get area.
	if (get_id_ == put_id_) {
		const auto n = epptr()-pbase();
		std::copy(pbase(), epptr(), eback());
		setg(eback(), gptr(), eback()+n);
	}
	return 0;
}

inline std::streamsize blockbuf::showmanyc() {
	// The largest contiguous read we support is whatever is left in the get area.
  return egptr() - gptr();
}

inline blockbuf::int_type blockbuf::underflow() {
	// Read the next block from block storage.
	const auto n = read(get_id_+1, eback(), 0);
	// If the read returned zero bytes, we're at the end of file.
	if (n == 0) {
    return traits_type::eof(); 
	}
	// Otherwise, increment the get pointer and reset the get area.
	++get_id_;			
  setg(eback(), eback(), eback()+n);
	// Return the first element in the new get area.
  return traits_type::to_int_type(get_area_[0]);
}

inline blockbuf::int_type blockbuf::uflow() {
	// Call underflow() to refresh the get area
	const auto res = underflow();
	// If underflow didn't return eof, bump the get pointer.
	if (res != traits_type::eof()) {
		gbump(1);
	}
	// Return the result of the call to underflow().
	return res;
}

inline std::streamsize blockbuf::xsgetn(char_type* s, std::streamsize count) {
  std::streamsize total = 0;
  while (total < count) {
		// Compute how much we have left to read
		const auto remaining = count-total;
		// Compute what's left in the get area, and underflow if we're out of characters
		auto available = showmanyc();
		if ((available == 0) && (underflow() == traits_type::eof())) {
			break;
		}
		available = showmanyc();
		// Read whichever is smaller, what's available or what's remaining to be read
    const auto chunk_size = std::min(available, remaining);
    std::copy(gptr(), gptr()+chunk_size, s+total);
    total += chunk_size;
		// Bump the get pointer
		gbump(chunk_size);
  } 
  return total;
}

inline std::streamsize blockbuf::xsputn(const char_type* s, std::streamsize count) {
  std::streamsize total = 0;
  while (total < count) {
		// Compute how much we have left to write
		const auto remaining = count-total;
		// Compute what's left in the put area, and overflow if we're out of characters
		auto available = epptr()-pptr();
		if ((available == 0) && (extend_put_area(remaining) == -1) && (bump_put_area(remaining) == -1)) {
			break;
		}
		available = epptr()-pptr();
		// Write whichever is smaller, what's available or what's remaining to be written
    const auto chunk_size = std::min(available, count-total);
    std::copy(s+total, s+total+chunk_size, pptr());
    total += chunk_size;
		// Bump the get pointer
		pbump(chunk_size);
  } 
  return total;
}

inline blockbuf::int_type blockbuf::overflow(int_type c) {
	// If we can't extend the put area or bump the put area, return eof to signal error
	if ((extend_put_area(1) == -1) && (bump_put_area(1) == -1)) {
		return traits_type::eof();
	}
	// Nothing else to do for an eof character.
  if (c == traits_type::eof()) {
    return traits_type::to_int_type('0');
  }
	// Otherwise, write this char to the put area and return success.
	*pptr() = c;
  return traits_type::to_int_type(c);
}

inline blockbuf::int_type blockbuf::pbackfail(int_type c) {
	// Does nothing.
	// TODO(eschkufz): Page back to the previous block and keep producing characters
  return traits_type::eof();
}

inline int blockbuf::extend_put_area(size_t n) {
	// We can't do anything if the put area spans an entire block.
	const auto capacity = blocksize_ - (epptr()-pbase());
	if (capacity == 0) {
		return -1;
	}
	// Extend by what's left or the request, whichever is smaller.
	const auto current = pptr()-pbase();
	const auto extension = std::min(n, capacity);
	setp(pbase(), pbase()+current+extension);
	pbump(current);
	return 0;
}

inline int blockbuf::bump_put_area(size_t n) {
	// Sync the put area.
	if (sync() == -1) {
		return -1;
	}
	// Reset pointers
	++put_id_;
	const auto extension = std::min(n, blocksize_);
	setp(pbase(), pbase()+extension);
	return 0;
}

} // namespace graft

#endif
