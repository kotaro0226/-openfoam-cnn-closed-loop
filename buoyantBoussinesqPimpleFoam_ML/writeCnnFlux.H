{
    typedef int (*MlCnnPredictBlocksFn)(const char*, const float*, float*, std::int64_t, char*, std::size_t);

    static void* cnnBridgeHandle = nullptr;
    static MlCnnPredictBlocksFn cnnPredictBlocks = nullptr;
    static bool reportedNoModel = false;
    static bool wroteFaceCorrespondence = false;

    const char* cnnModelEnv = std::getenv("ML_CNN_MODEL");
    const bool enableCnnFlux = cnnModelEnv != nullptr && word(cnnModelEnv) != "0";

    if (enableCnnFlux)
    {
        const char* writeTimingEnv = std::getenv("ML_WRITE_TIMING");
        const bool writeMlTiming =
            writeTimingEnv != nullptr && word(writeTimingEnv) != "0";
        const auto mlCnnFluxStart = std::chrono::steady_clock::now();
        double mlCnnPredictMs = 0.0;

        label cnnInferenceInterval = 1;
        const char* intervalEnv = std::getenv("ML_CNN_INFERENCE_INTERVAL");
        if (intervalEnv != nullptr && word(intervalEnv) != "0")
        {
            cnnInferenceInterval = max(label(1), label(readLabel(IStringStream(intervalEnv)())));
        }

        const bool haveCachedCnnFlux =
            mesh.foundObject<surfaceScalarField>("pEqnFlux_cnn")
         && mesh.foundObject<surfaceScalarField>("pEqnFlux_cnn_mask");
        const bool shouldUpdateCnnFlux =
            !haveCachedCnnFlux || (runTime.timeIndex() % cnnInferenceInterval == 0);

        if (!shouldUpdateCnnFlux)
        {
            if (writeMlTiming)
            {
                Info<< "[ML_TIMING] time=" << runTime.timeName()
                    << " cnn_flux_reused=1"
                    << " inferenceInterval=" << cnnInferenceInterval << nl << endl;
            }
        }
        else
        {

        const std::string cnnModelPath(cnnModelEnv);
        if (cnnPredictBlocks == nullptr)
        {
            const char* bridgeEnv = std::getenv("ML_CNN_BRIDGE");
            const std::string bridgePath =
                bridgeEnv != nullptr && word(bridgeEnv) != "0"
              ? std::string(bridgeEnv)
              : std::string("libcnn_libtorch_bridge.so");

            Info<< "[ML] Loading CNN LibTorch bridge: " << bridgePath << nl << endl;
            cnnBridgeHandle = dlopen(bridgePath.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (cnnBridgeHandle == nullptr)
            {
                FatalErrorInFunction
                    << "Could not load ML_CNN_BRIDGE: " << bridgePath
                    << " dlerror=" << dlerror() << exit(FatalError);
            }

            void* symbol = dlsym(cnnBridgeHandle, "ml_cnn_predict_blocks");
            if (symbol == nullptr)
            {
                FatalErrorInFunction
                    << "Could not resolve ml_cnn_predict_blocks in " << bridgePath
                    << " dlerror=" << dlerror() << exit(FatalError);
            }
            cnnPredictBlocks = reinterpret_cast<MlCnnPredictBlocksFn>(symbol);

            Info<< "[ML] CNN TorchScript model path: " << cnnModelPath << nl << endl;
        }

        const label nCells = mesh.nCells();
        const label nInternalFaces = mesh.nInternalFaces();
        const vectorField& cellCentres = mesh.C();
        const vectorField& faceCentres = mesh.Cf();
        const scalarField& rAUfValues = rAUf.primitiveField();
        const scalarField& phiHbyAValues = phiHbyA.primitiveField();
        const scalarField& divPhiHbyAValues = divPhiHbyA_f.primitiveField();

        const scalar GREAT = std::numeric_limits<scalar>::max();
        vector minC(GREAT, GREAT, GREAT);
        vector maxC(-GREAT, -GREAT, -GREAT);
        forAll(cellCentres, cellI)
        {
            minC.x() = min(minC.x(), cellCentres[cellI].x());
            minC.y() = min(minC.y(), cellCentres[cellI].y());
            minC.z() = min(minC.z(), cellCentres[cellI].z());
            maxC.x() = max(maxC.x(), cellCentres[cellI].x());
            maxC.y() = max(maxC.y(), cellCentres[cellI].y());
            maxC.z() = max(maxC.z(), cellCentres[cellI].z());
        }

        const label nPerDim = label(std::round(std::cbrt(scalar(nCells))));
        if (nPerDim*nPerDim*nPerDim != nCells)
        {
            FatalErrorInFunction
                << "CNN flux probe currently expects a structured cubic mesh. nCells="
                << nCells << exit(FatalError);
        }

        const vector spacing
        (
            (maxC.x() - minC.x())/scalar(nPerDim - 1),
            (maxC.y() - minC.y())/scalar(nPerDim - 1),
            (maxC.z() - minC.z())/scalar(nPerDim - 1)
        );

        const label validDim = nPerDim - 2;
        const label channels = 18;
        const label blockCore = 16;
        const label halo = 2;
        const label inputSize = blockCore + 2*halo;

        std::vector<label> ijkToCell(nCells, -1);
        std::vector<label> cellFaces(nCells*6, -1);
        std::vector<scalar> faceCanonicalSign(nInternalFaces, 1.0);
        std::vector<float> xValid(validDim*validDim*validDim*channels, 0.0f);

        auto fullLinear = [=](const label i, const label j, const label k)
        {
            return (i*nPerDim + j)*nPerDim + k;
        };
        auto validLinear = [=](const label i, const label j, const label k, const label ch)
        {
            return (((i*validDim + j)*validDim + k)*channels + ch);
        };

        forAll(cellCentres, cellI)
        {
            const vector c = cellCentres[cellI];
            const label i = label(std::round((c.x() - minC.x())/spacing.x()));
            const label j = label(std::round((c.y() - minC.y())/spacing.y()));
            const label k = label(std::round((c.z() - minC.z())/spacing.z()));
            if (i < 0 || i >= nPerDim || j < 0 || j >= nPerDim || k < 0 || k >= nPerDim)
            {
                FatalErrorInFunction << "Cell ijk out of bounds for cell " << cellI << exit(FatalError);
            }
            ijkToCell[fullLinear(i, j, k)] = cellI;

            scalar channelValues[18] = {};
            bool haveFace[6] = {};
            const cell& cFaces = mesh.cells()[cellI];
            forAll(cFaces, localFaceI)
            {
                const label faceI = cFaces[localFaceI];
                if (faceI >= nInternalFaces)
                {
                    continue;
                }

                const vector d = faceCentres[faceI] - c;
                const scalar ax = mag(d.x());
                const scalar ay = mag(d.y());
                const scalar az = mag(d.z());

                label faceSlot = -1;
                if (ax >= ay && ax >= az)
                {
                    faceSlot = d.x() < 0 ? 0 : 1;
                }
                else if (ay >= ax && ay >= az)
                {
                    faceSlot = d.y() < 0 ? 2 : 3;
                }
                else
                {
                    faceSlot = d.z() < 0 ? 4 : 5;
                }

                const label ch = 3*faceSlot;
                channelValues[ch] = rAUfValues[faceI];
                channelValues[ch + 1] = phiHbyAValues[faceI];
                channelValues[ch + 2] = divPhiHbyAValues[faceI];
                haveFace[faceSlot] = true;
                cellFaces[cellI*6 + faceSlot] = faceI;
            }

            if (i > 0 && i < nPerDim - 1 && j > 0 && j < nPerDim - 1 && k > 0 && k < nPerDim - 1)
            {
                for (label faceSlot = 0; faceSlot < 6; ++faceSlot)
                {
                    if (!haveFace[faceSlot])
                    {
                        FatalErrorInFunction
                            << "Interior cell missing internal face slot. cell=" << cellI
                            << " slot=" << faceSlot << exit(FatalError);
                    }
                }
                const label vi = i - 1;
                const label vj = j - 1;
                const label vk = k - 1;
                for (label ch = 0; ch < channels; ++ch)
                {
                    xValid[validLinear(vi, vj, vk, ch)] = float(channelValues[ch]);
                }
            }
        }

        for (label idx = 0; idx < nCells; ++idx)
        {
            if (ijkToCell[idx] < 0)
            {
                FatalErrorInFunction << "Structured ijk-to-cell map has missing entries" << exit(FatalError);
            }
        }

        const labelUList& owner = mesh.owner();
        const labelUList& neighbour = mesh.neighbour();
        label nNegativeCanonicalFaces = 0;
        for (label faceI = 0; faceI < nInternalFaces; ++faceI)
        {
            const vector d = cellCentres[neighbour[faceI]] - cellCentres[owner[faceI]];
            const scalar ax = mag(d.x());
            const scalar ay = mag(d.y());
            const scalar az = mag(d.z());

            scalar sign = 1.0;
            if (ax >= ay && ax >= az)
            {
                sign = d.x() < 0 ? -1.0 : 1.0;
            }
            else if (ay >= ax && ay >= az)
            {
                sign = d.y() < 0 ? -1.0 : 1.0;
            }
            else
            {
                sign = d.z() < 0 ? -1.0 : 1.0;
            }

            faceCanonicalSign[faceI] = sign;
            if (sign < 0)
            {
                ++nNegativeCanonicalFaces;
            }
        }

        const char* writeFaceCorrespondenceEnv = std::getenv("ML_WRITE_FACE_CORRESPONDENCE");
        const bool writeFaceCorrespondence =
            writeFaceCorrespondenceEnv != nullptr
         && word(writeFaceCorrespondenceEnv) != "0"
         && !wroteFaceCorrespondence;

        if (writeFaceCorrespondence)
        {
            label maxDiagCells = 16;
            const char* maxDiagCellsEnv = std::getenv("ML_FACE_CORRESPONDENCE_MAX_CELLS");
            if (maxDiagCellsEnv != nullptr && word(maxDiagCellsEnv) != "0")
            {
                maxDiagCells = max(label(1), label(readLabel(IStringStream(maxDiagCellsEnv)())));
            }

            fileName diagDir
            (
                runTime.path()
              / "postProcessing"
              / "cnnFaceCorrespondence"
              / runTime.timeName()
            );
            mkDir(diagDir);

            const fileName diagPath(diagDir/"face_correspondence.csv");
            std::ofstream diagFile(diagPath.c_str());
            diagFile
                << "cell_id,i,j,k,slot,slot_name,face_id,owner,neighbour,"
                << "cell_owner_neighbour_sign,canonical_face_sign,"
                << "face_center_x,face_center_y,face_center_z,"
                << "cell_center_x,cell_center_y,cell_center_z\n";

            const char* slotNames[6] = {"xm", "xp", "ym", "yp", "zm", "zp"};
            label nDiagCells = 0;
            forAll(cellCentres, cellI)
            {
                const vector c = cellCentres[cellI];
                const label i = label(std::round((c.x() - minC.x())/spacing.x()));
                const label j = label(std::round((c.y() - minC.y())/spacing.y()));
                const label k = label(std::round((c.z() - minC.z())/spacing.z()));

                if (!(i > 0 && i < nPerDim - 1 && j > 0 && j < nPerDim - 1 && k > 0 && k < nPerDim - 1))
                {
                    continue;
                }

                bool haveAllFaces = true;
                for (label faceSlot = 0; faceSlot < 6; ++faceSlot)
                {
                    if (cellFaces[cellI*6 + faceSlot] < 0)
                    {
                        haveAllFaces = false;
                        break;
                    }
                }
                if (!haveAllFaces)
                {
                    continue;
                }

                for (label faceSlot = 0; faceSlot < 6; ++faceSlot)
                {
                    const label faceI = cellFaces[cellI*6 + faceSlot];
                    const label ownerI = owner[faceI];
                    const label neighbourI = neighbour[faceI];
                    const label cellOwnerNeighbourSign =
                        ownerI == cellI ? 1 : (neighbourI == cellI ? -1 : 0);
                    const vector fc = faceCentres[faceI];

                    diagFile
                        << cellI << ','
                        << i << ',' << j << ',' << k << ','
                        << faceSlot << ',' << slotNames[faceSlot] << ','
                        << faceI << ','
                        << ownerI << ',' << neighbourI << ','
                        << cellOwnerNeighbourSign << ','
                        << faceCanonicalSign[faceI] << ','
                        << fc.x() << ',' << fc.y() << ',' << fc.z() << ','
                        << c.x() << ',' << c.y() << ',' << c.z() << '\n';
                }

                ++nDiagCells;
                if (nDiagCells >= maxDiagCells)
                {
                    break;
                }
            }

            wroteFaceCorrespondence = true;
            Info<< "[ML_FACE_MAP] wrote face correspondence diagnostic: "
                << diagPath << " cells=" << nDiagCells
                << " faceOrder=xm,xp,ym,yp,zm,zp" << nl << endl;
        }

        forAll(cellCentres, cellI)
        {
            if (cellFaces[cellI*6] < 0)
            {
                continue;
            }

            const vector c = cellCentres[cellI];
            const label i = label(std::round((c.x() - minC.x())/spacing.x()));
            const label j = label(std::round((c.y() - minC.y())/spacing.y()));
            const label k = label(std::round((c.z() - minC.z())/spacing.z()));
            if (i > 0 && i < nPerDim - 1 && j > 0 && j < nPerDim - 1 && k > 0 && k < nPerDim - 1)
            {
                const label vi = i - 1;
                const label vj = j - 1;
                const label vk = k - 1;
                for (label faceSlot = 0; faceSlot < 6; ++faceSlot)
                {
                    const label faceI = cellFaces[cellI*6 + faceSlot];
                    const label ch = 3*faceSlot;
                    xValid[validLinear(vi, vj, vk, ch + 1)] =
                        float(faceCanonicalSign[faceI]*xValid[validLinear(vi, vj, vk, ch + 1)]);
                    xValid[validLinear(vi, vj, vk, ch + 2)] =
                        float(faceCanonicalSign[faceI]*xValid[validLinear(vi, vj, vk, ch + 2)]);
                }
            }
        }

        std::vector<label> starts;
        for (label s = halo; s <= validDim - halo - blockCore; s += blockCore)
        {
            starts.push_back(s);
        }
        const label maxStart = validDim - halo - blockCore;
        if (starts.empty() || starts.back() != maxStart)
        {
            starts.push_back(maxStart);
        }

        std::vector<label> blockStartI;
        std::vector<label> blockStartJ;
        std::vector<label> blockStartK;
        for (const label bi : starts)
        {
            for (const label bj : starts)
            {
                for (const label bk : starts)
                {
                    blockStartI.push_back(bi);
                    blockStartJ.push_back(bj);
                    blockStartK.push_back(bk);
                }
            }
        }

        const label nBlocks = blockStartI.size();
        const label inputBlockValues = channels*inputSize*inputSize*inputSize;
        const label outputBlockValues = 6*blockCore*blockCore*blockCore;
        std::vector<float> inputValues(nBlocks*inputBlockValues, 0.0f);
        std::vector<float> outputValues(nBlocks*outputBlockValues, 0.0f);

        for (label blockI = 0; blockI < nBlocks; ++blockI)
        {
            const label bi = blockStartI[blockI];
            const label bj = blockStartJ[blockI];
            const label bk = blockStartK[blockI];
            const label blockOffset = blockI*inputBlockValues;

            for (label ch = 0; ch < channels; ++ch)
            {
                for (label di = 0; di < inputSize; ++di)
                {
                    for (label dj = 0; dj < inputSize; ++dj)
                    {
                        for (label dk = 0; dk < inputSize; ++dk)
                        {
                            const label vi = bi - halo + di;
                            const label vj = bj - halo + dj;
                            const label vk = bk - halo + dk;
                            const label outIdx =
                                blockOffset + (((ch*inputSize + di)*inputSize + dj)*inputSize + dk);
                            inputValues[outIdx] = xValid[validLinear(vi, vj, vk, ch)];
                        }
                    }
                }
            }
        }

        char errorBuf[4096] = {};
        int predictStatus = 0;
        {
            sigFpe::ignore ignoreFpe;
            const auto mlPredictStart = std::chrono::steady_clock::now();
            predictStatus = cnnPredictBlocks
            (
                cnnModelPath.c_str(),
                inputValues.data(),
                outputValues.data(),
                static_cast<std::int64_t>(nBlocks),
                errorBuf,
                sizeof(errorBuf)
            );
            const auto mlPredictEnd = std::chrono::steady_clock::now();
            mlCnnPredictMs +=
                std::chrono::duration<double, std::milli>(mlPredictEnd - mlPredictStart).count();
        }

        if (predictStatus != 0)
        {
            FatalErrorInFunction
                << "CNN batch inference failed with status " << predictStatus
                << ": " << errorBuf << exit(FatalError);
        }

        std::vector<scalar> faceAccum(nInternalFaces, 0.0);
        std::vector<label> faceCount(nInternalFaces, 0);

        for (label blockI = 0; blockI < nBlocks; ++blockI)
        {
            const label bi = blockStartI[blockI];
            const label bj = blockStartJ[blockI];
            const label bk = blockStartK[blockI];
            const float* y = outputValues.data() + blockI*outputBlockValues;

            for (label faceSlot = 0; faceSlot < 6; ++faceSlot)
            {
                for (label di = 0; di < blockCore; ++di)
                {
                    for (label dj = 0; dj < blockCore; ++dj)
                    {
                        for (label dk = 0; dk < blockCore; ++dk)
                        {
                            const label vi = bi + di;
                            const label vj = bj + dj;
                            const label vk = bk + dk;
                            const label cellI = ijkToCell[fullLinear(vi + 1, vj + 1, vk + 1)];
                            const label faceI = cellFaces[cellI*6 + faceSlot];
                            if (faceI >= 0 && faceI < nInternalFaces)
                            {
                                const label outIdx =
                                    (((faceSlot*blockCore + di)*blockCore + dj)*blockCore + dk);
                                faceAccum[faceI] += faceCanonicalSign[faceI]*scalar(y[outIdx]);
                                ++faceCount[faceI];
                            }
                        }
                    }
                }
            }
        }

        if (!mesh.foundObject<surfaceScalarField>("pEqnFlux_cnn"))
        {
            mesh.objectRegistry::store
            (
                new surfaceScalarField
                (
                    IOobject
                    (
                        "pEqnFlux_cnn",
                        runTime.timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    phiHbyA
                )
            );
        }

        surfaceScalarField& pEqnFluxCnn =
            mesh.lookupObjectRef<surfaceScalarField>("pEqnFlux_cnn");

        if (!mesh.foundObject<surfaceScalarField>("pEqnFlux_cnn_mask"))
        {
            mesh.objectRegistry::store
            (
                new surfaceScalarField
                (
                    IOobject
                    (
                        "pEqnFlux_cnn_mask",
                        runTime.timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    mesh,
                    dimensionedScalar("zero", dimless, 0.0)
                )
            );
        }

        surfaceScalarField& pEqnFluxCnnMask =
            mesh.lookupObjectRef<surfaceScalarField>("pEqnFlux_cnn_mask");

        pEqnFluxCnn = dimensionedScalar("zero", phiHbyA.dimensions(), 0.0);
        pEqnFluxCnnMask = dimensionedScalar("zero", dimless, 0.0);

        scalarField& cnnInternal = pEqnFluxCnn.primitiveFieldRef();
        scalarField& cnnMaskInternal = pEqnFluxCnnMask.primitiveFieldRef();
        label nPredictedFaces = 0;
        for (label faceI = 0; faceI < nInternalFaces; ++faceI)
        {
            if (faceCount[faceI] > 0)
            {
                cnnInternal[faceI] = faceAccum[faceI]/scalar(faceCount[faceI]);
                cnnMaskInternal[faceI] = 1.0;
                ++nPredictedFaces;
            }
        }

        pEqnFluxCnn.correctBoundaryConditions();
        pEqnFluxCnnMask.correctBoundaryConditions();

        const char* writeCnnFluxFieldsEnv = std::getenv("ML_WRITE_CNN_FLUX_FIELDS");
        const bool writeCnnFluxFields =
            writeCnnFluxFieldsEnv && Foam::string(writeCnnFluxFieldsEnv) == "1";

        if (writeCnnFluxFields)
        {
            pEqnFluxCnn.write();
            pEqnFluxCnnMask.write();
        }

        Info<< "[ML] CNN pEqnFlux probe: predictedInternalFaces="
            << nPredictedFaces << "/" << nInternalFaces
            << " negativeCanonicalFaces=" << nNegativeCanonicalFaces
            << " nBlocks=" << nBlocks
            << " inferenceInterval=" << cnnInferenceInterval
            << " model=" << cnnModelPath << nl << endl;

        if (writeMlTiming)
        {
            const auto mlCnnFluxEnd = std::chrono::steady_clock::now();
            const double mlCnnFluxMs =
                std::chrono::duration<double, std::milli>(mlCnnFluxEnd - mlCnnFluxStart).count();
            Info<< "[ML_TIMING] time=" << runTime.timeName()
                << " cnn_flux_total_ms=" << mlCnnFluxMs
                << " cnn_predict_ms=" << mlCnnPredictMs << nl << endl;
        }
        }
    }
    else if (!reportedNoModel)
    {
        Info<< "[ML] ML_CNN_MODEL is not set; skipping solver-side CNN flux probe" << nl << endl;
        reportedNoModel = true;
    }
}
