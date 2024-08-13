/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt and NOTICES.txt for license information
 ************************************************************************/

#include "mscclpp/mscclpp_nccl.h"

std::unordered_map<ncclUniqueId, ncclUniqueId> mscclpp_uniqueIdMap;
