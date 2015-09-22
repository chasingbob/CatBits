namespace Algorithms

module NeuralNetwork =

    let computeSigmoid (inputs : float seq) (weights : float seq) : float =
        let inputList = inputs |> List.ofSeq
        let weightsList = weights |> List.ofSeq
        if inputList.Length <> weightsList.Length 
            then invalidArg "inputs" (sprintf "Inputs and weights counts should be the same length")
        List.zip inputList weightsList
        |> List.sumBy (fun iw -> (fst iw) * (snd iw))
        //|> tanh

        

