#pragma once

class Formatter {
public:
	static std::string toString(cudaExtent const& extents) {
		std::stringstream res;
		res << "Extents[depth: ";
		res << extents.depth;
		res << ", width:" << extents.width;
		res << ", height: " << extents.height;
		res << "]";
		return res.str();

	}

	static std::string toString(cudaChannelFormatDesc const& desc) {
		std::stringstream res;
		res << "ChannelDesc[F:";
		res << (desc.f == cudaChannelFormatKindFloat
			? "Float"
			: desc.f == cudaChannelFormatKindUnsigned
			? "Unsigned"
			: desc.f == cudaChannelFormatKindSigned
			? "Signed"
			: desc.f == cudaChannelFormatKindNone
			? "None"
			: "Unknown");
		res << "," << desc.w << "," << desc.x << "," << desc.y << "," << desc.z;
		res << "]";
		return res.str();
	}

	static std::string toString(std::vector<float> const& vec) {
		std::stringstream res;
		res << "vec [";
		for (auto& elem : vec) res << elem << ",";
		res << "]";
		return res.str();
	}

	static std::string toString(nv::Dims dims) {
		std::stringstream res;
		res << "Num Dims: ";
		res << dims.nbDims;
		res << "[";

		for (int i = 0; i < dims.nbDims; i++) {
			res << dims.d[i];
			if (i != dims.nbDims - 1) res << ",";
		}
		res << "]";
		return res.str();
	}
};
