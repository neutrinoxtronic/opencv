#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{
   struct GAPI_EXPORTS_W_SIMPLE GCompileArg { };

   GAPI_EXPORTS_W GCompileArgs compile_args(gapi::GKernelPackage pkg);
   GAPI_EXPORTS_W GCompileArgs compile_args(gapi::GNetPackage pkg);

   class GAPI_EXPORTS_W_SIMPLE GProtoArg { };
   class GAPI_EXPORTS_W_SIMPLE GProtoInputArgs { };
   class GAPI_EXPORTS_W_SIMPLE GProtoOutputArgs { };
   class GAPI_EXPORTS_W_SIMPLE GRunArg {  };

   using GProtoInputArgs  = GIOProtoArgs<In_Tag>;
   using GProtoOutputArgs = GIOProtoArgs<Out_Tag>;

   namespace gapi
   {
       GAPI_EXPORTS_W gapi::GNetPackage networks(cv::gapi::ie::GenParams params);

       namespace ie
       {
           class GAPI_EXPORTS_W_SIMPLE GenParams { };
       }  // namespace ie
   }  // namespace gapi
}  // namespace cv
