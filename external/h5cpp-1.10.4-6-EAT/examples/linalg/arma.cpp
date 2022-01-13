#include <armadillo>
#include <h5cpp/all>


int main(){
	{ // CREATE - WRITE
		arma::mat M(2,3); M.ones();				            // create a matrix
		h5::fd_t fd = h5::create("arma.h5",H5F_ACC_TRUNC); 	// and a file
		h5::ds_t ds = h5::create<short>(fd,"create then write"
				,h5::current_dims{10,20}
				,h5::max_dims{10,H5S_UNLIMITED}
				,h5::chunk{2,3} | h5::fill_value<short>{3} |  h5::gzip{9}
		);
		h5::write( ds,  M, h5::offset{2,2}, h5::stride{1,3}  );
	}
	{
		arma::vec V( {1.,2.,3.,4.,5.,6.,7.,8.}); 	// create a vector
		// simple one shot write that computes current dimensions and saves matrix
		h5::write( "arma.h5", "one shot create write",  V);
		// what if you want to position a matrix inside a higher dimension with some added complexity?	
		h5::write( "arma.h5", "arma vec inside matrix",  V // object contains 'count' and rank being written
			,h5::current_dims{40,50}  // control file_space directly where you want to place vector
			,h5::offset{5,0}            // when no explicit current dimension given current dimension := offset .+ object_dim .* stride (hadamard product)  
 			,h5::count{1,1}
			,h5::stride{3,5}
			,h5::block{2,4}
			,h5::max_dims{40,H5S_UNLIMITED}  // wouldn't it be nice to have unlimited dimension? if no explicit chunk is set, then the object dimension 
							 // is used as unit chunk
		);
	}
	{ // CREATE - READ: we're reading back the dataset created in the very first step
	  // note that data is only data, can be reshaped, cast to any format and content be modified through filtering 
		auto fd = h5::open("arma.h5", H5F_ACC_RDWR,           // you can have multiple fd open with H5F_ACC_RDONLY, but single write
				h5::fclose_degree_strong | h5::sec2); 		   // and able to set various properties  
		h5::ds_t ds = h5::create<float>(fd,"dataset", h5::current_dims{3,2}, h5::fill_value<float>(NAN));  // create dataset, default to NaN-s
		auto M  = h5::read<arma::mat>( fd,"dataset" ); 				   // read data back as matrix
		M.print();
	}
	{ // READ: 
		arma::mat M = h5::read<arma::mat>("arma.h5","create then write"); // read entire dataset back with a single read
		M.print();
	}
}


