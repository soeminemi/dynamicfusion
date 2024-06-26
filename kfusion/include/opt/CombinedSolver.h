#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cudaUtil.h>
#include <SolverIteration.h>
#include <CombinedSolverParameters.h>
#include <CombinedSolverBase.h>
#include <OptGraph.h>
#include <cuda_profiler_api.h>
#include <macro_utils.hpp>
#include <kfusion/warp_field.hpp>
class CombinedSolver : public CombinedSolverBase
{

public:
    CombinedSolver(kfusion::WarpField *warpField, CombinedSolverParameters params)
    {
        m_combinedSolverParameters = params;
        m_warp = warpField;
    }

    void initializeProblemInstance(const std::vector<cv::Vec3f> &canonical_vertices,
                                   const std::vector<cv::Vec3f> &canonical_normals,
                                   const std::vector<cv::Vec3f> &live_vertices,
                                   const std::vector<cv::Vec3f> &live_normals)
    {
        m_canonicalVerticesOpenCV = canonical_vertices;
        m_canonicalNormalsOpenCV = canonical_normals;
        m_liveVerticesOpenCV = live_vertices;
        m_liveNormalsOpenCV = live_normals;


        unsigned int D = m_warp->getNodes()->size();
        unsigned int N = canonical_vertices.size();

        m_dims = { D, N };
        std::cout<<"initial warp dims"<<std::endl;
        m_rotationDeform    = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
        m_translationDeform = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

        m_canonicalVerticesOpt = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
        m_liveVerticesOpt      = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

        m_canonicalNormalsOpt  = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
        m_liveNormalsOpt       = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

        m_weights              = createEmptyOptImage({N}, OptImage::Type::FLOAT, KNN_NEIGHBOURS, OptImage::GPU, true);
        std::cout<<"reset gpu memory"<<std::endl;
        resetGPUMemory();
        std::cout<<"init connectivity"<<std::endl;
        initializeConnectivity(m_canonicalVerticesOpenCV);

#ifdef SOLVER_PATH
        std::cout<<"solver path: "<<m_solverInfo.size()<<","<<std::string(TOSTRING(SOLVER_PATH))<<std::endl;
        if(m_solverInfo.size() == 0)
        {
            std::string solver_file = std::string(TOSTRING(SOLVER_PATH)) + "dynamicfusion.t";
            std::cout<<"add solvers"<<std::endl;
            addOptSolvers(m_dims, solver_file);
            std::cout<<"solver added"<<std::endl;
        }
#else
        std::cerr<<"Please define a path for your solvers."<<std::endl;
        exit(-1);
#endif
        std::cout<<"end initial"<<std::endl;
    }
    void initializeConnectivity(const std::vector<cv::Vec3f> canonical_vertices)
    {
        unsigned int N = (unsigned int) canonical_vertices.size();

        std::vector<std::vector<int> > graph_vector(KNN_NEIGHBOURS + 1, vector<int>(N));
//        std::vector<float> weights(N * KNN_NEIGHBOURS);
        std::vector<float[KNN_NEIGHBOURS]> weights(N);
//FIXME: KNN doesn't need to be recomputed every time.
        for(int count = 0; count < canonical_vertices.size(); count++)
        {
            graph_vector[0].push_back(count);
            m_warp->getWeightsAndUpdateKNN(canonical_vertices[count], weights[count]);
            for(int i = 1; i < graph_vector.size(); i++)
                graph_vector[i].push_back((int)m_warp->getRetIndex()->at(i-1));
        }
        m_weights->update(weights);
        m_data_graph = std::make_shared<OptGraph>(graph_vector);

    }

    virtual void combinedSolveInit() override
    {
        m_functionTolerance = 1e-6f;
        m_paramTolerance = 1e-5f;

        m_problemParams.set("RotationDeform", m_rotationDeform);
        m_problemParams.set("TranslationDeform", m_translationDeform);

        m_problemParams.set("CanonicalVertices", m_canonicalVerticesOpt);
        m_problemParams.set("LiveVertices", m_liveVerticesOpt);

        m_problemParams.set("CanonicalNormals", m_canonicalNormalsOpt);
        m_problemParams.set("LiveNormals", m_liveNormalsOpt);

        m_problemParams.set("Weights", m_weights);

        m_problemParams.set("DataG", m_data_graph);
//        m_problemParams.set("RegG", m_reg_graph);

        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
        m_solverParams.set("function_tolerance", &m_functionTolerance);
//        m_solverParams.set("max_trust_region_radius", &m_trust_region_radius);
//        m_solverParams.set("q_tolerance", &m_paramTolerance);
    }

    virtual void preSingleSolve() override {
//        resetGPUMemory();
    }
    virtual void postSingleSolve() override {
        copyResultToCPUFromFloat3();
    }

    virtual void preNonlinearSolve(int) override {}

    virtual void postNonlinearSolve(int) override {}

    virtual void combinedSolveFinalize() override {
        reportFinalCosts("Robust Mesh Deformation", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), nan(""));
    }

    void resetGPUMemory()
    {
        uint N = (uint)m_canonicalVerticesOpenCV.size();
        std::vector<float3> h_canonical_vertices(N);
        std::vector<float3> h_canonical_normals(N);
        std::vector<float3> h_live_vertices(N);
        std::vector<float3> h_live_normals(N);

        for(int i = 0; i < N; i++)
        {
//            FIXME: this code could look better
            if(std::isnan(m_canonicalVerticesOpenCV[i][0]) ||
               std::isnan(m_canonicalVerticesOpenCV[i][1]) ||
               std::isnan(m_canonicalVerticesOpenCV[i][2])) continue;

            if(std::isnan(m_canonicalNormalsOpenCV[i][0]) ||
               std::isnan(m_canonicalNormalsOpenCV[i][1]) ||
               std::isnan(m_canonicalNormalsOpenCV[i][2])) continue;

            if(std::isnan(m_liveVerticesOpenCV[i][0]) ||
               std::isnan(m_liveVerticesOpenCV[i][1]) ||
               std::isnan(m_liveVerticesOpenCV[i][2])) continue;

            if(std::isnan(m_liveNormalsOpenCV[i][0]) ||
               std::isnan(m_liveNormalsOpenCV[i][1]) ||
               std::isnan(m_liveNormalsOpenCV[i][2])) continue;


            h_canonical_vertices[i] = make_float3(m_canonicalVerticesOpenCV[i][0], m_canonicalVerticesOpenCV[i][1], m_canonicalVerticesOpenCV[i][2]);
            h_canonical_normals[i] = make_float3(m_canonicalNormalsOpenCV[i][0], m_canonicalNormalsOpenCV[i][1], m_canonicalNormalsOpenCV[i][2]);
            h_live_vertices[i] = make_float3(m_liveVerticesOpenCV[i][0], m_liveVerticesOpenCV[i][1], m_liveVerticesOpenCV[i][2]);
            h_live_normals[i] = make_float3(m_liveNormalsOpenCV[i][0], m_liveNormalsOpenCV[i][1], m_liveNormalsOpenCV[i][2]);
        }
        m_canonicalVerticesOpt->update(h_canonical_vertices);
        m_canonicalNormalsOpt->update(h_canonical_normals);
        m_liveVerticesOpt->update(h_live_vertices);
        m_liveNormalsOpt->update(h_live_normals);

        uint D = (uint)m_warp->getNodes()->size();
        std::vector<float3> h_translation(D);
        std::vector<float3> h_rotation(D);

        for(int i = 0; i < m_warp->getNodes()->size(); i++)
        {
            float x,y,z;
            auto t = m_warp->getNodes()->at(i).transform;
            t.getTranslation(x,y,z);
            h_translation[i] = make_float3(x,y,z);

            t.getRotation().getRodrigues(x,y,z);
            h_rotation[i] = make_float3(x,y,z);
        }

        m_rotationDeform->update(h_rotation);
        m_translationDeform->update(h_translation);
    }

    std::vector<cv::Vec3f> result()
    {
        return m_resultVertices;
    }

    void copyResultToCPUFromFloat3()
    {
        unsigned int N = (unsigned int)m_warp->getNodes()->size();
        std::vector<float3> h_translation(N);
        m_translationDeform->copyTo(h_translation);

        for (unsigned int i = 0; i < N; i++)
            m_warp->getNodes()->at(i).transform.encodeTranslation(h_translation[i].x, h_translation[i].y, h_translation[i].z);
    }

private:

    kfusion::WarpField *m_warp;

    ml::Timer m_timer;

    // Current index in solve
    std::vector<unsigned int> m_dims;

    std::shared_ptr<OptImage> m_rotationDeform;
    std::shared_ptr<OptImage> m_translationDeform;

    std::shared_ptr<OptImage> m_canonicalVerticesOpt;
    std::shared_ptr<OptImage> m_liveVerticesOpt;
    std::shared_ptr<OptImage> m_canonicalNormalsOpt;
    std::shared_ptr<OptImage> m_liveNormalsOpt;
    std::shared_ptr<OptImage> m_weights;
    std::shared_ptr<OptGraph> m_reg_graph;
    std::shared_ptr<OptGraph> m_data_graph;

    std::vector<cv::Vec3f> m_canonicalVerticesOpenCV;
    std::vector<cv::Vec3f> m_canonicalNormalsOpenCV;
    std::vector<cv::Vec3f> m_liveVerticesOpenCV;
    std::vector<cv::Vec3f> m_liveNormalsOpenCV;
    std::vector<cv::Vec3f> m_resultVertices;


    float m_functionTolerance;
    float m_paramTolerance;
    float m_trust_region_radius;
};

