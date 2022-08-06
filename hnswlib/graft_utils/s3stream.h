#ifndef GRAFT_S3STREAM_H 
#define GRAFT_S3STREAM_H

#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <sstream>
#include <string>

#include "blockbuf.h"

// This class provides a c++ stream interface to s3 objects. The main complication
// here is that s3 doesn't support appends. As a result, we treat an s3 directory
// as a block device where a file is spread out across multiple blocks.

namespace graft {

class s3buf : public blockbuf {
  public:
    // Constructors:
    explicit s3buf(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object);
    ~s3buf() override;

  private:
		static const size_t BLOCK_SIZE = 32 * 1024 * 1024; // 32MB

		size_t device_end() override;
		void device_clear() override;
		size_t block_capacity() override;
		size_t block_size(size_t block_id) override;

		int read(size_t block_id, char_type* buffer, size_t offset) override;
		int write(size_t block_id, char_type* buffer, size_t n) override; 

		// Return a path to an object in the root bucket which corresponds to this block.
		std::string get_key(size_t block_id) const;

		// Helper methods for tracking device end.
		size_t read_cached_device_end();
		void set_cached_device_end(size_t id);
		void write_cached_device_end();

		// S3 Client
		Aws::S3::S3Client& client_;

		// The path to the object directory to use as a backing store.
		std::string object_;

		// A cached copy of the contents of block 0
		size_t device_end_;

		// Partial request objects
		Aws::S3::Model::PutObjectRequest put_request_;
		Aws::S3::Model::GetObjectRequest get_request_;
};

class is3stream : public std::istream {
  public:
    is3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object);
    ~is3stream() override;

  private:
    s3buf buf_;
};

class os3stream : public std::ostream {
  public:
    os3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object);
    ~os3stream() override;

  private:
    s3buf buf_;
};

class s3stream : public std::iostream {
  public:
    s3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object);
    ~s3stream() override;

  private:
    s3buf buf_;
};

inline s3buf::s3buf(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object) :
		blockbuf(new char_type[BLOCK_SIZE], new char_type[BLOCK_SIZE]), client_(client), object_(object) {
	// Initialize request objects
	put_request_.SetBucket(bucket);	
  get_request_.SetBucket(bucket);
	// Initialize device_end to sentinel value
	device_end_ = 0;
}

inline s3buf::~s3buf() {
	// Free get/put areas.
	delete[] get_area_;
	delete[] put_area_;
	// Write back device end
	write_cached_device_end();
}

inline size_t s3buf::device_end() {
	return read_cached_device_end();
}

inline void s3buf::device_clear() {
	set_cached_device_end(1);
}

inline size_t s3buf::block_capacity() {
	return BLOCK_SIZE;
}

inline size_t s3buf::block_size(size_t block_id) {
	std::cout << "BLOCK SIZE " << block_id << std::endl;
	return read(block_id, nullptr, block_capacity()+1);
}

inline int s3buf::read(size_t block_id, char_type* buffer, size_t offset) {
	std::cout << "READ " << block_id << std::endl;
	// Open backing file for this block (id+1)
	get_request_.SetKey(get_key(block_id+1));
	auto res = client_.GetObject(get_request_);
	// Reading from a block which was never written is undefined, return 0.
	if (!res.IsSuccess()) {
		return 0;
	}
	// Read block size
  auto& ss = res.GetResult().GetBody();
	size_t block_size = 0;
	ss.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
	// Only try reading if we asked for an offset within the block_size
	if (offset < block_size) {
		ss.seekg(offset, std::ios::cur);
		ss.read(buffer+offset, block_size-offset);
	}
	return block_size;
}

inline int s3buf::write(size_t block_id, char_type* buffer, size_t n) {
	std::cout << "WRITE " << block_id << std::endl;
	// Open backing file for this block (id+1).
	put_request_.SetKey(get_key(block_id+1));
	// Write blocksize and new data.
	const auto mode = std::ios::binary | std::ios::in | std::ios::out;
  std::shared_ptr<Aws::StringStream> ss = Aws::MakeShared<Aws::StringStream>("SampleAllocationTag", mode);
	ss->write(reinterpret_cast<char*>(&n), sizeof(n));
	ss->write(buffer, n);
	put_request_.SetBody(ss);
  client_.PutObject(put_request_);
	// Update device end if necessary
	if (block_id >= read_cached_device_end()) {
		set_cached_device_end(block_id+1);
	}
	// Return the size of this block
	return n;
}

inline std::string s3buf::get_key(size_t block_id) const {
	std::ostringstream oss;
	oss << object_ << "/" << block_id;
	return oss.str();
}

inline size_t s3buf::read_cached_device_end() {
	// Check block zero only if device_end_ is the sentinel value.
	if (device_end_ == 0) {
		get_request_.SetKey(get_key(0));
		auto res = client_.GetObject(get_request_);
		if (!res.IsSuccess()) {
			device_end_ = 1;
		} 
		else {
			auto& ss = res.GetResult().GetBody();
			ss.read(reinterpret_cast<char*>(&device_end_), sizeof(device_end_));
		}
	}
	return device_end_;
}


inline void s3buf::set_cached_device_end(size_t id) {
	device_end_ = id;
}

inline void s3buf::write_cached_device_end() {
	// Write block zero only if device_end_ is the sentinel value.
	if (device_end_ != 0) {
		put_request_.SetKey(get_key(0));
		const auto mode = std::ios::binary | std::ios::in | std::ios::out;
		std::shared_ptr<Aws::StringStream> ss = Aws::MakeShared<Aws::StringStream>("SampleAllocationTag", mode);
		ss->write(reinterpret_cast<char*>(&device_end_), sizeof(device_end_));
		put_request_.SetBody(ss);
		client_.PutObject(put_request_);
	}
}

inline is3stream::is3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object) : 
		std::istream(&buf_), buf_(client, bucket, object) {
	buf_.open(std::ios_base::in | std::ios_base::app);
}

inline is3stream::~is3stream() {
	buf_.close();
}

inline os3stream::os3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object) : 
		std::ostream(&buf_), buf_(client, bucket, object) {
	buf_.open(std::ios_base::out | std::ios_base::trunc);
}

inline os3stream::~os3stream() {
	buf_.close();
}

inline s3stream::s3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object) : 
		std::iostream(&buf_), buf_(client, bucket, object) {
	buf_.open(std::ios_base::in | std::ios_base::out | std::ios_base::app);
}

inline s3stream::~s3stream() {
	buf_.close();
}

} // namespace graft

#endif
