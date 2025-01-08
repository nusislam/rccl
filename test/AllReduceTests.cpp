/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"
#include <rccl/rccl.h>
#include <iostream>

namespace RcclUnitTesting
{
  TEST(AllReduce, OutOfPlace)
  {
    printf("A\n");
    TestBed testBed;
    printf("B\n");
    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 384};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, OutOfPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat16, ncclFloat64, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMin};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {12888};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, InPlace)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclInt8};
    std::vector<ncclRedOp_t>    const redOps          = {ncclProd};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {384};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, InPlaceGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclInt32, ncclFp8E4M3, ncclFp8E5M2};
    std::vector<ncclRedOp_t>    const redOps          = {ncclMax};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {393216, 12888, 384};
    std::vector<bool>           const inPlaceList     = {true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, ManagedMem)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclUint8, ncclUint64};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {2500};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, Channels)
  {
    TestBed testBed;
    if(testBed.ev.maxGpus >= 8) {
      if(testBed.ev.isGfx94) {
        // Configuration
        std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
        std::vector<ncclDataType_t> const dataTypes       = {ncclBfloat16};
        std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
        std::vector<int>            const roots           = {0};
        std::vector<int>            const numElements     = {64 * 1024 * 1024, 1024};
        std::vector<bool>           const inPlaceList     = {false};
        std::vector<bool>           const managedMemList  = {false};
        std::vector<bool>           const useHipGraphList = {false, true};
        std::vector<const char *>   const channelList     = {"84", "112"};
        bool                        const enableSweep     = false; 
        for (auto channel : channelList) {
          setenv("NCCL_MIN_NCHANNELS", channel, 1);
          testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                                inPlaceList, managedMemList, useHipGraphList, enableSweep);
          testBed.Finalize();
          unsetenv("NCCL_MIN_NCHANNELS");
        }
      }
    }
  }

  TEST(AllReduce, ManagedMemGraph)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat64, ncclBfloat16};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {4314};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {true};
    std::vector<bool>           const useHipGraphList = {true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();
  }

  TEST(AllReduce, DISABLED_Clique)
  {
    // Set clique env var prior to TestBed
    setenv("RCCL_ENABLE_CLIQUE", "1", 1);

    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = testBed.GetAllSupportedDataTypes();
    std::vector<ncclRedOp_t>    const redOps          = testBed.GetAllSupportedRedOps();
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1048576, 1024};
    std::vector<bool>           const inPlaceList     = {false, true};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false, true};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList);
    testBed.Finalize();

    unsetenv("RCCL_ENABLE_CLIQUE");
  }

  // This tests using custom pre-mult scalars reductions
  TEST(AllReduce, PreMultScalar)
  {
    TestBed testBed;

    // Configuration
    ncclFunc_t                  const  funcType      = ncclCollAllReduce;
    std::vector<ncclDataType_t> const& dataTypes     = {ncclFloat32};
    ncclRedOp_t                 const  redOp         = ncclSum;
    std::vector<int>            const  numElements   = {384 * 1024, 384 * 32, 384};
    bool                        const  inPlace       = false;
    bool                        const  useManagedMem = false;

    OptionalColArgs options;

    // Terminate the test as soon as first failure occurs
    bool isCorrect = true;
    for (int totalRanks : testBed.ev.GetNumGpusList())
    for (int isMultiProcess : testBed.ev.GetIsMultiProcessList())
    {
      int const numProcesses = isMultiProcess ? totalRanks : 1;
      const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
      testBed.InitComms(TestBed::GetDeviceIdsList(numProcesses, totalRanks, gpuPriorityOrder));

      for (int dataIdx = 0; dataIdx < dataTypes.size() && isCorrect; ++dataIdx)
      {
        ncclDataType_t const dataType = dataTypes[dataIdx];

        // Set scalars per rank
        PtrUnion scalarsPerRank;
        scalarsPerRank.AllocateCpuMem(totalRanks * DataTypeToBytes(dataType));
        for (int i = 0;  i < totalRanks; i++)
        {
          double F = i;
          scalarsPerRank.Set(dataType, i, i, F);
        }
        int const numBytes = totalRanks * DataTypeToBytes(dataType);
        memcpy(options.scalarTransport.ptr, scalarsPerRank.ptr, numBytes);

        // Test various scalar residence modes
        for (int scalarMode = 0; scalarMode <= 1 && isCorrect; ++scalarMode)
        {
          if (testBed.ev.showNames)
            INFO("%s %d-ranks AllReduce (custom-scalar Mode %d %s)\n",
                 isMultiProcess ? "MP" : "SP",
                 totalRanks, scalarMode, ncclDataTypeNames[dataType]);

          for (int i = 0; i < numElements.size() && isCorrect; ++i)
          {
            //options.scalarMode = scalarMode;
            options.redOp = redOp;
            testBed.SetCollectiveArgs(funcType, dataType,
                                      numElements[i], numElements[i],
                                      options);
            // For performance, only allocate and prepare data on largest size
            if (i == 0)
            {
              testBed.AllocateMem(inPlace, useManagedMem);
              testBed.PrepareData();
            }
            testBed.ExecuteCollectives();
            testBed.ValidateResults(isCorrect);
          }
          testBed.DeallocateMem();
        }
      }
      testBed.DestroyComms();
    }
    testBed.Finalize();
  }

  /*TEST(AllReduce, UserBufferRegister)
  {
    TestBed testBed;

    // Configuration
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes       = {ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps          = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1024};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};
    
    bool                        const  useManagedMem   = false;
    bool                        const  userRegistered  = true;

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlaceList, managedMemList, useHipGraphList, userRegistered);

    testBed.Finalize();
  }*/

  TEST(AllReduce, UBR)
  {
    TestBed testBed;
    std::vector<ncclFunc_t>     const funcTypes       = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const tmpDataTypes    = {ncclFloat32};
    std::vector<ncclRedOp_t>    const tmpRedOps       = {ncclSum};
    std::vector<int>            const roots           = {0};
    std::vector<int>            const numElements     = {1024};
    std::vector<bool>           const inPlaceList     = {false};
    std::vector<bool>           const managedMemList  = {false};
    std::vector<bool>           const useHipGraphList = {false};

    bool                        const  useManagedMem   = false;
    bool                        const  userRegistered  = false;

    std::vector<int> sortedN = numElements;
    std::sort(sortedN.rbegin(), sortedN.rend());
    OptionalColArgs optionalArgs;
    // Filter out any unsupported datatypes, in case only subset has been compiled for

    std::vector<ncclDataType_t> const& supportedDataTypes = testBed.GetAllSupportedDataTypes();
    std::vector<ncclDataType_t> dataTypes;

    for (auto dt : tmpDataTypes)
    {
      for (int i = 0; i < supportedDataTypes.size(); ++i)
      {
        if (supportedDataTypes[i] == dt)
        {
          dataTypes.push_back(dt);
          break;
        }
      }
    }

    bool isCorrect = true;
    std::vector<ncclRedOp_t> const& supportedOps = testBed.GetAllSupportedRedOps();
    std::vector<ncclRedOp_t> redOps;

    for (auto redop : tmpRedOps)
    {
      for (int i = 0; i < supportedOps.size(); ++i)
      {
        if (supportedOps[i] == redop)
        {
          redOps.push_back(redop);
          break;
        }
      }
    }

    int numGpus = 2;//testBed.ev.GetNumGpusList().back();    
    int isMultiProcess = 1; //mscclpp not enabled if 0
    int ranksPerGpu = 1;
    int const numChildren = isMultiProcess ? numGpus : 1;
    int const numRanks    = numGpus*ranksPerGpu;
    const std::vector<int>& gpuPriorityOrder = testBed.ev.GetGpuPriorityOrder();
    testBed.InitComms(testBed.GetDeviceIdsList(numChildren, numGpus, ranksPerGpu, gpuPriorityOrder));

    if (testing::Test::HasFailure())
    {
      isCorrect = false;
      printf("TESTING HAS FAILURE -1\n");
    }

    for (int ftIdx = 0; ftIdx < funcTypes.size()      && isCorrect; ++ftIdx)
    for (int dtIdx = 0; dtIdx < dataTypes.size()      && isCorrect; ++dtIdx)
    for (int rdIdx = 0; rdIdx < redOps.size()         && isCorrect; ++rdIdx)
    for (int rtIdx = 0; rtIdx < roots.size()          && isCorrect; ++rtIdx)
    for (int ipIdx = 0; ipIdx < inPlaceList.size()    && isCorrect; ++ipIdx)
    for (int mmIdx = 0; mmIdx < managedMemList.size() && isCorrect; ++mmIdx)
    {
      for (int neIdx = 0; neIdx < numElements.size() && isCorrect; ++neIdx)
      {
        int numInputElements, numOutputElements;
        CollectiveArgs::GetNumElementsForFuncType(funcTypes[ftIdx],
                                                    sortedN[neIdx],
                                                    numRanks,
                                                    &numInputElements,
                                                    &numOutputElements);
        optionalArgs.redOp = redOps[rdIdx];
        optionalArgs.root = roots[rtIdx];
        testBed.SetCollectiveArgs(funcTypes[ftIdx],
                                  dataTypes[dtIdx],
                                  numInputElements,
                                  numOutputElements,
                                  optionalArgs,
                                  4,
                                  0);
        if (testing::Test::HasFailure())
        {
          printf("TESTING HAS FAILURE 0\n");
          isCorrect = false;
          continue;
        }

        // Only allocate once for largest size
        if (neIdx == 0)
        {
          //ncclAllReduce ID is 4 based
          testBed.AllocateMem(inPlaceList[ipIdx], managedMemList[mmIdx], 0, 4, -1, userRegistered);
          if (testing::Test::HasFailure())
          {
            printf("TESTING HAS FAILURE A\n");
            isCorrect = false;
            continue;
          }
        }

        for (int hgIdx = 0; hgIdx < useHipGraphList.size() && isCorrect; ++hgIdx)
        {
          // There are some cases when data does not need to be re-prepared
          // e.g. AllReduce subarray expected results are still valid
          bool canSkip = (neIdx != 0 && !inPlaceList[ipIdx] &&
                            (funcTypes[ftIdx] == ncclCollBroadcast ||
                             funcTypes[ftIdx] == ncclCollReduce    ||
                             funcTypes[ftIdx] == ncclCollAllReduce));
          if (!canSkip) testBed.PrepareData(0, 4, -1);
          if (testing::Test::HasFailure())
          {
            printf("TESTING HAS FAILURE B\n");
            isCorrect = false;
            continue;
          }

          std::string name = testBed.GetTestCaseName(numGpus, isMultiProcess,
                                                     funcTypes[ftIdx], dataTypes[dtIdx],
                                                     redOps[rdIdx], roots[rtIdx],
                                                     inPlaceList[ipIdx], managedMemList[mmIdx],
                                                     useHipGraphList[hgIdx], ranksPerGpu);

          if (testBed.ev.showNames)
          {
            printf("%s [%9d elements]\n", name.c_str(), numInputElements);
          }

          std::vector<int> currentRanksEmpty = {};
          testBed.ExecuteCollectives(currentRanksEmpty, 0, useHipGraphList[hgIdx]);
          if (useHipGraphList[hgIdx]) {
            testBed.LaunchGraphs();
            testBed.DestroyGraphs();
          }
          
          if (testing::Test::HasFailure())
          {
            printf("TESTING HAS FAILURE C\n");
            isCorrect = false;
            continue;
          }
          testBed.ValidateResults(isCorrect, 0, 4, -1);
          if (!isCorrect)
          {
            ERROR("Incorrect output for %s\n", name.c_str());
          }
        }
      }
      testBed.DeallocateMem();
      //testBed.DeallocateMem(0, 4, -1);
    }

    testBed.DestroyComms();  
    testBed.Finalize();
  }

}
