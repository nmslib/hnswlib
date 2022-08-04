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
    explicit s3buf(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object, std::ios_base::openmode mode);
    ~s3buf() override;

  private:
		static const size_t BLOCK_SIZE = 1024;

		int read(size_t block_id, char_type* buffer, size_t offset) override;
		int write(size_t block_id, char_type* buffer, size_t n) override; 

		// Return a path to an object in the root bucket which corresponds to this block.
		std::string get_key(size_t block_id) const;

		// S3 Client
		Aws::S3::S3Client& client_;

		// The path to the object directory to use as a backing store.
		std::string object_;

		// Partial request objects
		Aws::S3::Model::PutObjectRequest put_request_;
		Aws::S3::Model::GetObjectRequest get_request_;
};

class is3stream : public std::istream {
  public:
    is3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object);
    ~is3stream() override = default;

  private:
    s3buf buf_;
};

class os3stream : public std::ostream {
  public:
    os3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object);
    ~os3stream() override = default;

  private:
    s3buf buf_;
};

class s3stream : public std::iostream {
  public:
    s3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object);
    ~s3stream() override = default;

  private:
    s3buf buf_;
};

inline s3buf::s3buf(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object, std::ios_base::openmode mode) :
		blockbuf(new char_type[BLOCK_SIZE], new char_type[BLOCK_SIZE], BLOCK_SIZE), 
		client_(client),
		object_(object) {
	// Initialize request objects
	put_request_.SetBucket(bucket);	
  get_request_.SetBucket(bucket);
	// Delete root object in truncate mode.
	if (mode & std::ios_base::trunc) {
		// TODO
	} 
}

inline s3buf::~s3buf() {
	delete[] get_area_;
	delete[] put_area_;
}

inline int s3buf::read(size_t block_id, char_type* buffer, size_t offset) {
	// Get path to backing file for this block.
	const auto key = get_key(block_id);
	get_request_.SetKey(key);
	// Try reading the file
	auto res = client_.GetObject(get_request_);
	// If the file doesn't exist the block doesn't exist and the size is zero.
	if (!res.IsSuccess()) {
		return 0;
	}
	// Read block size
  auto& content = res.GetResultWithOwnership().GetBody();
	size_t block_size = 0;
	content.read(reinterpret_cast<char*>(&block_size), sizeof(block_size));
	// Only try reading if we asked for an offset within the block_size
	if (offset < block_size) {
		content.seekg(offset, std::ios::cur);
		content.read(buffer+offset, block_size-offset);
	}
	return block_size;
}

inline int s3buf::write(size_t block_id, char_type* buffer, size_t n) {
	// Get path to backing file for this block.
	const auto key = get_key(block_id);
	put_request_.SetKey(key);
	// Write blocksize and new data.
	const auto mode = std::ios::binary | std::ios::in | std::ios::out;
  std::shared_ptr<Aws::StringStream> input_data = Aws::MakeShared<Aws::StringStream>("SampleAllocationTag", mode);
	input_data->write(reinterpret_cast<char*>(&n), sizeof(n));
	input_data->write(buffer, n);
	put_request_.SetBody(input_data);
	// Check results for failure, otherwise return block size
  const auto res = client_.PutObject(put_request_);
	return res.IsSuccess() ? n : -1;
}

inline std::string s3buf::get_key(size_t block_id) const {
	std::ostringstream oss;
	oss << object_ << "/" << block_id;
	return oss.str();
}

inline is3stream::is3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object) : 
	std::istream(&buf_), 
	buf_(client, bucket, object, std::ios_base::in | std::ios_base::app) { 
}

inline os3stream::os3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object) : 
	std::ostream(&buf_), 
	buf_(client, bucket, object, std::ios_base::out | std::ios_base::trunc) { 
}

inline s3stream::s3stream(Aws::S3::S3Client& client, const std::string& bucket, const std::string& object) : 
	std::iostream(&buf_), 
	buf_(client, bucket, object, std::ios_base::in | std::ios_base::out | std::ios_base::app) { 
}

} // namespace graft

#endif
