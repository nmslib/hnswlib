#pragma once
#include <string.h>
namespace hnswlib {
	template <typename dist_t> class BruteforceSearch : public AlgorithmInterface<dist_t> {
	public:
		BruteforceSearch(SpaceInterface<dist_t> *s) {

		}
		BruteforceSearch(SpaceInterface<dist_t> *s, size_t maxElements) {
			maxelements_ = maxElements;
			
			data_size_= s->get_data_size();
			fstdistfunc_ = s->get_dist_func();
			dist_func_param_ = s->get_dist_func_param();
			cout << data_size_ << "\n";
			size_per_element_ = data_size_ + sizeof(labeltype);
			data_=(char *)malloc(maxElements*size_per_element_);
			cur_element_count = 0;
		}
		~BruteforceSearch() {
			free(data_);
		}
		char *data_;
		size_t maxelements_;
		size_t cur_element_count;
		size_t size_per_element_;

		size_t data_size_;
		DISTFUNC<dist_t> fstdistfunc_;
		void *dist_func_param_;

		void addPoint(void *datapoint, labeltype label) {
			
			if (cur_element_count >= maxelements_)
			{
				cout << "The number of elements exceeds the specified limit\n";
				throw exception();
			};
			memcpy(data_ + size_per_element_*cur_element_count+ data_size_, &label, sizeof(labeltype));
			memcpy(data_ + size_per_element_*cur_element_count, datapoint, data_size_);
			cur_element_count++;
		};
		std::priority_queue< std::pair< dist_t, labeltype >> searchKnn(void *query_data,int k) {
			std::priority_queue< std::pair< dist_t, labeltype >> topResults;
			for (int i = 0; i < k; i++) {
				dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_*i, dist_func_param_);
				topResults.push(std::pair<dist_t, labeltype>(dist, *((labeltype*)(data_ + size_per_element_*i + data_size_))));
			}
			dist_t lastdist= topResults.top().first;
			for (int i = k; i < cur_element_count; i++) {
				dist_t dist=fstdistfunc_(query_data, data_ + size_per_element_*i, dist_func_param_);
				if(dist < lastdist) {				
					topResults.push(std::pair<dist_t, labeltype>(dist,*((labeltype*) (data_ + size_per_element_*i + data_size_))));
					if (topResults.size() > k)
						topResults.pop();
					lastdist = topResults.top().first;
				}

			}
			return topResults;
		};
	};
}
