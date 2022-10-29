#include "FoveatedSynthesis.h"
#include "../utils/Config.h"
#include "FovNeRF.h"
#include "Enhancement.h"
#include "ImageGen.h"

constexpr auto NUM_LAYERS = 3u;
constexpr auto STEREO_FOVEA_R = NUM_LAYERS;
constexpr auto NUM_NETS = 2u;

class FoveatedSynthesis_Impl
{
public:
	FoveatedSynthesis_Impl(const std::string &modelDir, sptr<Camera> cam,
						   const std::vector<sptr<Camera>> &layerCams, bool stereo);

	void run(View &view, glm::vec2 foveaPos, bool showPerf, glm::vec2 foveaPosR);

	GLuint getGlResultTexture(uint index) const;

	glm::vec2 getDepthRange() const;

private:
	bool _stereo;
	uint _nRays;
	uint _nSamples;
	glm::vec2 _depthRange;
	sptr<Camera> _fullCam;
	sptr<Camera> _cams[NUM_LAYERS];
	sptr<FovNeRFCore> _nets[NUM_NETS];
	sptr<FovNeRF> _infers[NUM_NETS];
	sptr<Enhancement> _enhancements[NUM_LAYERS];
	sptr<ImageGen> _imageGens[NUM_LAYERS + 1];
	sptr<CudaArray<glm::vec3>> _rays;
	sptr<CudaArray<glm::vec4>> _clrs;
	sptr<CudaArray<glm::vec4>> _imageData[NUM_LAYERS + 1];
};

FoveatedSynthesis_Impl::FoveatedSynthesis_Impl(const std::string &modelDir, sptr<Camera> cam,
											   const std::vector<sptr<Camera>> &layerCams,
											   bool stereo) : _fullCam(cam), _stereo(stereo)
{
	// Load nets
	for (uint i = 0; i < NUM_NETS; ++i)
		_nets[i].reset(new FovNeRFCore());
	_nets[0]->load(modelDir + "/fovea.trt");
	_nets[1]->load(modelDir + "/periph.trt");

	// Load configs
	Config foveaConfig(modelDir + "/fovea.ini");
	Config periphConfig(modelDir + "/periph.ini");
	uint nSamples[] = {
		foveaConfig.getInt("n_samples"),
		periphConfig.getInt("n_samples")};
	uint encodeDim = foveaConfig.getInt("xfreqs");
	uint coordChns = foveaConfig.getBool("with_radius") ? 3 : 2;
	_depthRange = {foveaConfig.getFloat("near"), foveaConfig.getFloat("far")};

	// Init cams
	for (uint i = 0; i < NUM_LAYERS; ++i)
		_cams[i] = layerCams[i];

	uint nRays[NUM_LAYERS];
	uint nTotRays = 0;
	for (uint i = 0; i < NUM_LAYERS; ++i)
		nTotRays += nRays[i] = _cams[i]->nRays();
	if (_stereo)
		nTotRays += nRays[0];

	// Init infers
	_infers[0].reset(new FovNeRF(_nets[0], nRays[0], nSamples[0],
								 _depthRange, encodeDim, coordChns));
	_infers[1].reset(new FovNeRF(_nets[1], nRays[1] + nRays[2], nSamples[1],
								 _depthRange, encodeDim, coordChns));

	// Init image gens
	for (uint i = 0; i < NUM_LAYERS; ++i)
		_imageGens[i].reset(new ImageGen(_cams[i]->res()));
	if (_stereo)
		_imageGens[STEREO_FOVEA_R].reset(new ImageGen(_cams[0]->res()));

	// Init enhancements
	glm::vec2 enhancementParams[] = {
		{3.0f, 0.2f}, {5.0f, 0.2f}, {5.0f, 0.2f}};
	for (uint i = 0; i < NUM_LAYERS; ++i)
		_enhancements[i].reset(new Enhancement(_cams[i]->res(), enhancementParams[i]));

	// Create buffers
	_rays.reset(new CudaArray<glm::vec3>(nTotRays));
	_clrs.reset(new CudaArray<glm::vec4>(nTotRays));
	for (uint i = 0; i < NUM_LAYERS; ++i)
		_imageData[i].reset(new CudaArray<glm::vec4>(_cams[i]->nPixels()));
	if (_stereo)
		_imageData[STEREO_FOVEA_R].reset(new CudaArray<glm::vec4>(_cams[0]->nPixels()));
}

void FoveatedSynthesis_Impl::run(View &view, glm::vec2 foveaPos, bool showPerf, glm::vec2 foveaPosR)
{
	CudaEvent eStart, eGenRays, eInferred, eGenImage, eEnhance;
	uint offset;

	cudaEventRecord(eStart);

	glm::vec2 foveaOffset(foveaPos - (glm::vec2)_fullCam->res() / 2.0f);
	foveaOffset /= _fullCam->f();
	glm::vec3 foveaOffset3(foveaOffset.x, foveaOffset.y, 0.0f);

	glm::vec2 foveaOffsetR(foveaPosR - (glm::vec2)_fullCam->res() / 2.0f);
	foveaOffsetR /= _fullCam->f();
	glm::vec3 foveaOffset3R(foveaOffsetR.x, foveaOffsetR.y, 0.0f);

	auto viewL = view.getStereoEye(0.06f, Eye_Left);
	auto viewR = view.getStereoEye(0.06f, Eye_Right);

	if (_stereo)
	{
		offset = 0;
		_cams[0]->getRays(sptr<CudaArray<glm::vec3>>(_rays->subArray(offset)), viewL, foveaOffset3);
		offset += _cams[0]->nRays();
		_cams[1]->getRays(sptr<CudaArray<glm::vec3>>(_rays->subArray(offset)), view, (foveaOffset3 + foveaOffset3R) / 2.0f);
		offset += _cams[1]->nRays();
		_cams[2]->getRays(sptr<CudaArray<glm::vec3>>(_rays->subArray(offset)), view, {});
		offset += _cams[2]->nRays();
		_cams[0]->getRays(sptr<CudaArray<glm::vec3>>(_rays->subArray(offset)), viewR, foveaOffset3R);
	}
	else
	{
		offset = 0;
		for (uint i = 0; i < NUM_LAYERS; ++i)
		{
			_cams[i]->getRays(sptr<CudaArray<glm::vec3>>(_rays->subArray(offset)),
							  view, i == NUM_LAYERS - 1 ? glm::vec3() : foveaOffset3);
			offset += _cams[i]->nRays();
		}
	}

	cudaEventRecord(eGenRays);

	if (_stereo)
	{
		offset = 0;
		_infers[0]->run(sptr<CudaArray<glm::vec4>>(_clrs->subArray(offset)),
						sptr<CudaArray<glm::vec3>>(_rays->subArray(offset)), viewL.t(), showPerf);
		offset += _infers[0]->nRays();
		_infers[1]->run(sptr<CudaArray<glm::vec4>>(_clrs->subArray(offset)),
						sptr<CudaArray<glm::vec3>>(_rays->subArray(offset)), view.t(), showPerf);
		offset += _infers[1]->nRays();
		_infers[0]->run(sptr<CudaArray<glm::vec4>>(_clrs->subArray(offset)),
						sptr<CudaArray<glm::vec3>>(_rays->subArray(offset)), viewR.t(), showPerf);
	}
	else
	{
		offset = 0;
		for (uint i = 0; i < NUM_NETS; ++i)
		{
			_infers[i]->run(sptr<CudaArray<glm::vec4>>(_clrs->subArray(offset)),
							sptr<CudaArray<glm::vec3>>(_rays->subArray(offset)), view.t(), showPerf);
			offset += _infers[i]->nRays();
		}
	}

	cudaEventRecord(eInferred);

	offset = 0;
	for (uint i = 0; i < NUM_LAYERS; ++i)
	{
		_cams[i]->restoreImage(_imageData[i], sptr<CudaArray<glm::vec4>>(_clrs->subArray(offset)));
		offset += _cams[i]->nRays();
	}
	if (_stereo)
		_cams[0]->restoreImage(_imageData[STEREO_FOVEA_R], sptr<CudaArray<glm::vec4>>(_clrs->subArray(offset)));

	cudaEventRecord(eGenImage);

	for (uint i = 0; i < NUM_LAYERS; ++i)
		_enhancements[i]->run(_imageData[i]);
	if (_stereo)
		_enhancements[0]->run(_imageData[STEREO_FOVEA_R]);

	cudaEventRecord(eEnhance);
	CHECK_EX(cudaDeviceSynchronize());

	for (uint i = 0; i < NUM_LAYERS; ++i)
		_imageGens[i]->run(_imageData[i]);
	if (_stereo)
		_imageGens[STEREO_FOVEA_R]->run(_imageData[STEREO_FOVEA_R]);

	float timeTotal, timeGenRays, timeInfer, timeGenImage, timeEnhance;
	cudaEventElapsedTime(&timeTotal, eStart, eGenImage);
	cudaEventElapsedTime(&timeGenRays, eStart, eGenRays);
	cudaEventElapsedTime(&timeInfer, eGenRays, eInferred);
	cudaEventElapsedTime(&timeGenImage, eInferred, eGenImage);
	cudaEventElapsedTime(&timeEnhance, eGenImage, eEnhance);
	if (showPerf)
	{
		std::ostringstream sout;
		sout << "Synthesis => Total: " << timeTotal << "ms (Gen rays: " << timeGenRays
			 << "ms, Infer: " << timeInfer << "ms, Gen image: " << timeGenImage
			 << "ms, Enhance: " << timeEnhance << "ms)";
		Logger::instance.info(sout.str().c_str());
	}
}

GLuint FoveatedSynthesis_Impl::getGlResultTexture(uint index) const
{
	return _imageGens[index]->getGlResultTexture();
}

glm::vec2 FoveatedSynthesis_Impl::getDepthRange() const
{
	return _depthRange;
}

FoveatedSynthesis::FoveatedSynthesis(const std::string &modelDir, sptr<Camera> cam,
									 const std::vector<sptr<Camera>> &layerCams, bool stereo)
	: _impl(new FoveatedSynthesis_Impl(modelDir, cam, layerCams, stereo))
{
}

void FoveatedSynthesis::run(View &view, glm::vec2 foveaPos, bool showPerf, glm::vec2 foveaPosR)
{
	_impl->run(view, foveaPos, showPerf, foveaPosR);
}

GLuint FoveatedSynthesis::getGlResultTexture(uint index) const
{
	return _impl->getGlResultTexture(index);
}

glm::vec2 FoveatedSynthesis::getDepthRange() const
{
	return _impl->getDepthRange();
}