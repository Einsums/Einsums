/*
 * Copyright (c) 2018 vargaconsulting, Toronto,ON Canada
 * Author: Varga, Steven <steven@vargaconsulting.ca>
 *
 */

#ifndef  H5CPP_STD_COUT
#define  H5CPP_STD_COUT




inline
std::ostream& operator<< (std::ostream& os, const h5::dxpl_t& dxpl) {
	os <<"handle: " << static_cast<hid_t>( dxpl );
#ifdef H5_HAVE_PARALLEL
	H5D_mpio_actual_io_mode_t io_mode;
	H5Pget_mpio_actual_io_mode( static_cast<hid_t>(dxpl), &io_mode);

		switch( io_mode ){
			case H5D_MPIO_NO_COLLECTIVE:
				os << "No collective I/O was performed. Collective I/O was not requested or collective I/O isn't possible on this dataset.";
				break;
			case H5D_MPIO_CHUNK_INDEPENDENT:
				os << "HDF5 performed one the chunk collective optimization schemes and each chunk was accessed independently.";
				break;
			case H5D_MPIO_CHUNK_COLLECTIVE:
				os << "HDF5 performed one the chunk collective optimization schemes and each chunk was accessed collectively";
				break;
			case H5D_MPIO_CHUNK_MIXED:
				os <<"HDF5 performed one the chunk collective optimization schemes and some chunks were accessed independently, some collectively";
				break;
			case H5D_MPIO_CONTIGUOUS_COLLECTIVE:
				os <<"Collective I/O was performed on a contiguous dataset.";
			   	break;
		}
#endif
    return os;
}



template <class T> inline
std::ostream& operator<<(std::ostream& os, const h5::impl::array<T>& arr){
	os << "{";
	if( arr.rank )
		// rank > 0 such as: vector,matrix,cube,...
		for(int i=0;i<arr.rank; i++){
			char sep = i != arr.rank - 1  ? ',' : '}';
			if( arr[i] < std::numeric_limits<hsize_t>::max() )
				os << arr[i] << sep;
			else
				os << "inf" << sep;
		}
	else // rank 0 : single values
		os << "n/a}";
	return os;
}

inline
std::ostream& operator<<(std::ostream &os, const h5::sp_t& sp) {
	//htri_t H5Sis_regular_hyperslab( hid_t space_id ) 1.10.0
	//herr_t H5Sget_select_bounds(hid_t space_id, hsize_t *start, hsize_t *end )
	// hssize_t H5Sget_select_npoints( hid_t space_id )
	hid_t id = static_cast<hid_t>( sp );
	#if H5_VERSION_GE(1,10,0)

	#endif
	h5::mute();

	h5::offset_t start,end;
	h5::current_dims_t current_dims;
	h5::max_dims_t max_dims;
	unsigned rank = h5::get_simple_extent_dims( sp, current_dims, max_dims);
	hsize_t total_elements = H5Sget_simple_extent_npoints( id );
 	herr_t err = H5Sget_select_bounds(id, *start, *end);
	start.rank = end.rank = rank;
	hsize_t nblocks =  H5Sget_select_hyper_nblocks( id );
	hsize_t ncoordinates = 2*rank*nblocks;

    os << "[rank]\t" << rank << "\t[total elements]\t" << total_elements << std::endl;
   	os << "[dimensions]\tcurrent: " << current_dims << "\tmaximum: " << max_dims << std::endl;
	os << "[selection]\tstart: " << start << "\tend:" << end << std::endl;

	h5::impl::unique_ptr<hsize_t> buffer{
			static_cast<hsize_t*>( std::calloc( ncoordinates, sizeof(hsize_t))) };
	if( H5Sget_select_hyper_blocklist(id, 0, nblocks, buffer.get() ) >= 0   ){
		os << "[selected block count]\t" << nblocks <<std::endl;
		os << "[selected blocks]\t";
		for( int i=0; i<nblocks; i++){
			os << "[{";
			for( int j=0; j<rank; j++) os << *( buffer.get() + i*2*rank+j ) << (j < rank-1 ? "," : "}{");
			for( int j=rank; j<2*rank; j++) os << *( buffer.get() + i*2*rank+j ) << ( j < 2*rank-1 ? "," : "}");
			os << "] ";
		}
	}
	h5::unmute();
	return os;
}

template <class T> inline
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec){
	os << "[";
	if(vec.size() < H5CPP_CONSOLE_WIDTH ){
		int i=0;
		for(; i<vec.size()-1; i++ ) os << vec[i] <<",";
		os << vec[i];
	}else{
		os << ".. fix me ..";
	}
	os << "]";
return os;
}


#endif

