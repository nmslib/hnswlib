#ifndef GRAFT_BLOCKBUF_H
#define GRAFT_BLOCKBUF_H

#include <iostream>
#include <streambuf>

// This class provides a c++ stream interface to a block storage device. The
// underlying abstraction to that device is that blocks are zero indexed
// and can be read/written atomically. 

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
    explicit blockbuf(char_type* get_area, char_type* put_area);
    virtual ~blockbuf() override = default;

		// API:
		void open(std::ios_base::openmode mode);
		void close();

	protected:
		// Block storage meta data interface:
		//
		// Points one past the last (0-indexed) block in use. 
		virtual size_t device_end() = 0;
		// Point device_end() back to 1.
		virtual void device_clear() = 0;
		// Get the maximum number of bytes a block can hold.
		virtual size_t block_capacity() = 0;
		// Get the number of bytes currently stored in a block.
		virtual size_t block_size(size_t block_id) = 0;

		// Block storage read/write interface:
		//
		// buffer[offset:] = block_id[offset:]. 
		//  - Return the number of bytes in the block, or -1 on error.
		//  - Reading from a block beyond num_blocks() is undefined.
		//  - Reading from a block which was never written is undefined.
		virtual int read(size_t block_id, char_type* buffer, size_t offset) = 0;
		// block_id[0:n] = buffer[0:n] .
		//  - Write n bytes to a block. This updates the size of the block in bytes.
		//  - Writes to a block beyond num_blocks() should update num_blocks().
		virtual int write(size_t block_id, char_type* buffer, size_t n) = 0; 

    // Get and Put areas (provided by the implementation)
    char_type* get_area_;
		char_type* put_area_;

		// Dirty bit for put area
		bool dirty_;

  private:
		// Get and Put area ids
		int get_id_;
		int put_id_;

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
		// Push back the end of the put area to make min(block_size, n) bytes available.
		int extend_put_area(size_t n);
		// Sync the put area, move to the next block id, and make min(block_size, n) available.
		int bump_put_area(size_t n);
};

inline blockbuf::blockbuf(char_type* get_area, char_type* put_area) : get_area_(get_area), put_area_(put_area) {
	// TODO(eschkufz): Open in a benign mode.
}

inline void blockbuf::open(std::ios_base::openmode mode) {
	// Trunc mode: 
	// Set num blocks to zero, get and put areas to fault to block 0 on first read/write
	if (mode & std::ios_base::trunc) {
		device_clear();
		get_id_ = -1;
		setg(get_area_, get_area_, get_area_);
		put_id_ = 0;
		setp(put_area_, put_area_);
	} 
	// Append mode: 
	// Position write head at end of last block, get area to fault on first read.
	if (mode & std::ios_base::app) {
		get_id_ = -1;
		setg(get_area_, get_area_, get_area_);
		put_id_ = device_end() - 1;
		const auto n = read(put_id_, put_area_, 0);
		setp(put_area_, put_area_+n);
		pbump(n);
	}
	// Regardless of mode, we start clean
	dirty_ = false;
}

inline void blockbuf::close() {
	pubsync();
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
	// If the dirty bit isn't set there's nothing to do.
	if (!dirty_) {
		return 0;
	}
	// If the put area is empty, there's nothing to do.
	const auto size = pptr()-pbase();
	if (size == 0) {
		dirty_ = false;
		return 0;
	}
	// If we're not at the end of the put area, read back what we're missing.
	if (pptr() != epptr()) {
		const auto capacity = read(put_id_, put_area_, size);
		if (capacity == -1) {
			return -1;
		}
		setp(pbase(), pbase()+capacity);
		pbump(size);
	}
	// Write back 
	if (write(put_id_, put_area_, size) == -1) {
		return -1;
	}
	// If get and put areas refer to the same block, copy the put area to the get area.
	if (get_id_ == put_id_) {
		const auto capacity = epptr()-pbase();
		std::copy(pbase(), epptr(), eback());
		setg(eback(), gptr(), eback()+capacity);
	}
	dirty_ = false;
	return 0;
}

inline std::streamsize blockbuf::showmanyc() {
	// The largest contiguous read we support is whatever is left in the get area.
  return egptr() - gptr();
}

inline blockbuf::int_type blockbuf::underflow() {
	// If this is the last block, we're at the end of file
	if (get_id_+1 == device_end()) {
    return traits_type::eof(); 
	}
	// Otherwise, read in the next block and update the get area
	++get_id_;			
	const auto capacity = read(get_id_, eback(), 0);
  setg(eback(), eback(), eback()+capacity);
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
	const auto available = block_capacity() - (epptr()-pbase());
	if (available == 0) {
		return -1;
	}
	// Extend by what's left or the request, whichever is smaller, and set dirty bit.
	const auto size = pptr()-pbase();
	const auto additional = std::min(n, available);
	setp(pbase(), pbase()+size+additional);
	pbump(size);
	dirty_ = true;
	return 0;
}

inline int blockbuf::bump_put_area(size_t n) {
	// Sync the put area.
	if (sync() == -1) {
		return -1;
	}
	// Reset pointers and set dirty bit
	++put_id_;
	const auto additional = std::min(n, block_capacity());
	setp(pbase(), pbase()+additional);
	dirty_ = true;
	return 0;
}

} // namespace graft

#endif
