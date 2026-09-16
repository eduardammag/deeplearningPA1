from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .preprocessing import create_boundary_mask, normalize_image

# DEFINIÇÃO DO DATASET
class SyntheticSegmentationDataset(Dataset):
    """
    Dataset personalizado para imagens sintéticas de segmentação.
    Cada amostra do dataset possui:

    DADO	        O QUE REPRESENTA 
    image	        A imagem que entra na rede
    semantic_mask	Qual classe cada pixel pertence
    instance_mask	Qual objeto individual cada pixel pertence
    boundary_mask	Onde estão as fronteiras entre objetos
    """


    # CONSTRUTOR
    def __init__(
        self,
        root_dir,
        boundary_width=1  #Largura da região considerada fronteira entre diferentes objetos.
    ):
        self.root_dir = Path(root_dir)
        # Guarda a largura da fronteira para utilizar
        # posteriormente na criação da boundary_mask.
        self.boundary_width = boundary_width


        # ENCONTRANDO AS AMOSTRAS
        self.samples = sorted(
            [
                path
                for path in self.root_dir.iterdir()
                if path.is_dir()
            ]
        )


        # O que está acontecendo aqui?
        #
        # self.root_dir.iterdir()
        #
        # percorre tudo que existe dentro da pasta principal.
        #
        # Imagine que temos:
        #
        # data/train/
        # ├── sample_001/
        # ├── sample_002/
        # ├── sample_003/
        # ├── arquivo.txt
        # └── outra_coisa.png
        #
        # O código:
        #
        #     if path.is_dir()
        #
        # seleciona SOMENTE as pastas.
        #
        # Portanto:
        #
        # self.samples =
        #
        # [
        #     data/train/sample_001,
        #     data/train/sample_002,
        #     data/train/sample_003
        # ]
        #
        # sorted(...)
        #
        # coloca essas pastas em ordem.
        #
        # Assim, o Dataset sabe quais são todas as amostras
        # disponíveis.


    # TAMANHO DO DATASET
    def __len__(self):
        """
        Retorna o número total de amostras do dataset.
        """

        # self.samples contém uma lista com todas as pastas
        # de amostras.
        #
        # len(self.samples) informa quantas existem.
        return len(self.samples)


    # ========================================================
    # OBTENDO UMA AMOSTRA
    # ========================================================

    def __getitem__(self, index):
        """
        Retorna uma amostra específica do dataset.

        Por exemplo:

            dataset[0]

        chama:

            __getitem__(0)

        e retorna a primeira amostra.
        """


        # ----------------------------------------------------
        # LOCALIZA A PASTA DA AMOSTRA
        # ----------------------------------------------------

        # self.samples[index] pega a pasta correspondente
        # ao índice solicitado.
        #
        # Se index = 0:
        #
        # sample_dir = data/train/sample_001
        #
        # Se index = 1:
        #
        # sample_dir = data/train/sample_002
        sample_dir = self.samples[index]


        # ----------------------------------------------------
        # DEFINIÇÃO DOS CAMINHOS DOS ARQUIVOS
        # ----------------------------------------------------

        # Dentro de cada pasta de amostra esperamos encontrar:
        #
        # image.png
        # semantic_mask.png
        # instance_mask.png

        image_path = sample_dir / "image.png"

        semantic_path = sample_dir / "semantic_mask.png"

        instance_path = sample_dir / "instance_mask.png"


        # ====================================================
        # CARREGANDO A IMAGEM
        # ====================================================

        # Image.open() abre a imagem usando PIL.
        #
        # np.array() transforma a imagem PIL em um array NumPy.
        #
        # Por exemplo, uma imagem RGB de tamanho 256x256
        # pode virar:
        #
        #     shape = (256, 256, 3)
        #
        # onde:
        #
        # 256 -> altura
        # 256 -> largura
        # 3   -> canais RGB
        image = np.array(
            Image.open(image_path)
        )


        # ====================================================
        # CARREGANDO A MÁSCARA SEMÂNTICA
        # ====================================================

        # Fazemos a mesma coisa com a máscara semântica.
        #
        # Essa máscara normalmente informa a CLASSE
        # de cada pixel.
        #
        # Por exemplo:
        #
        # 0 = background
        # 1 = carro
        # 2 = pessoa
        #
        # A máscara pode ter shape:
        #
        # (256, 256)
        #
        # porque cada pixel possui apenas um número
        # representando sua classe.
        semantic_mask = np.array(
            Image.open(semantic_path)
        )


        # ====================================================
        # CARREGANDO A MÁSCARA DE INSTÂNCIA
        # ====================================================

        # A instance_mask identifica objetos individuais.
        #
        # Por exemplo:
        #
        # 0 = background
        # 1 = objeto 1
        # 2 = objeto 2
        # 3 = objeto 3
        #
        # Mesmo que os objetos 1, 2 e 3 sejam da mesma classe,
        # eles recebem identificadores diferentes.
        instance_mask = np.array(
            Image.open(instance_path)
        )


        # ====================================================
        # CONVERTENDO A IMAGEM PARA TENSOR
        # ====================================================

        # np.array -> Tensor PyTorch
        #
        # torch.from_numpy()
        #
        # transforma o array NumPy em um tensor.
        image = torch.from_numpy(
            image
        ).float()


        # .float()
        #
        # converte os valores para ponto flutuante.
        #
        # Isso é importante porque redes neurais normalmente
        # trabalham com tensores do tipo float32 para as
        # imagens de entrada.
        #
        # Exemplo:
        #
        # antes:
        # uint8
        #
        # depois:
        # float32


        # ====================================================
        # CONVERTENDO A MÁSCARA SEMÂNTICA
        # ====================================================

        semantic_mask = torch.from_numpy(
            semantic_mask
        ).long()


        # A máscara semântica é convertida para long.
        #
        # Isso é importante porque máscaras de classes
        # normalmente representam categorias inteiras.
        #
        # Por exemplo:
        #
        #     0 -> background
        #     1 -> pessoa
        #     2 -> carro
        #
        # Em funções de perda como CrossEntropyLoss,
        # os targets normalmente precisam ser inteiros
        # do tipo LongTensor.


        # ====================================================
        # CONVERTENDO A MÁSCARA DE INSTÂNCIA
        # ====================================================

        instance_mask = torch.from_numpy(
            instance_mask
        ).long()


        # Novamente utilizamos long porque os identificadores
        # das instâncias são números inteiros.
        #
        # Por exemplo:
        #
        #     0 -> background
        #     1 -> objeto 1
        #     2 -> objeto 2
        #     3 -> objeto 3


        # ====================================================
        # NORMALIZAÇÃO DA IMAGEM
        # ====================================================

        # Aqui a imagem passa pela função:
        #
        #     normalize_image()
        #
        # Essa função foi criada no arquivo preprocessing.py.
        #
        # Ela provavelmente transforma os valores dos pixels
        # para uma escala adequada para a rede neural.
        #
        # Por exemplo, uma imagem originalmente com:
        #
        #     0 até 255
        #
        # pode ser transformada para:
        #
        #     0.0 até 1.0
        #
        # dependendo de como normalize_image foi implementada.
        image = normalize_image(image)


        # ====================================================
        # ORGANIZAÇÃO DOS CANAIS DA IMAGEM
        # ====================================================

        # Agora verificamos quantas dimensões o tensor
        # da imagem possui.
        if image.ndim == 2:


            # ------------------------------------------------
            # CASO A IMAGEM SEJA GRAYSCALE
            # ------------------------------------------------

            # Uma imagem grayscale normalmente possui:
            #
            #     altura x largura
            #
            # Exemplo:
            #
            #     (256, 256)
            #
            # Porém, o PyTorch normalmente espera imagens
            # no formato:
            #
            #     canais x altura x largura
            #
            # Portanto precisamos adicionar uma dimensão
            # para representar o canal.
            image = image.unsqueeze(0)


            # unsqueeze(0) adiciona uma dimensão na posição 0.
            #
            # Antes:
            #
            #     (256, 256)
            #
            # Depois:
            #
            #     (1, 256, 256)
            #
            # O 1 representa:
            #
            #     1 canal -> grayscale


        else:


            # ------------------------------------------------
            # CASO A IMAGEM SEJA COLORIDA
            # ------------------------------------------------

            # Uma imagem RGB normalmente chega do PIL/NumPy
            # no formato:
            #
            #     altura x largura x canais
            #
            # Exemplo:
            #
            #     (256, 256, 3)
            #
            # Entretanto, o PyTorch utiliza:
            #
            #     canais x altura x largura
            #
            # Portanto fazemos:
            #
            #     (H, W, C)
            #
            # virar:
            #
            #     (C, H, W)
            image = image.permute(2, 0, 1)


            # Exemplo:
            #
            # Antes:
            #
            #     (256, 256, 3)
            #
            # Depois:
            #
            #     (3, 256, 256)
            #
            # Agora a imagem está no formato esperado
            # normalmente pelas redes convolucionais do PyTorch.


        # ====================================================
        # CRIAÇÃO DA MÁSCARA DE FRONTEIRA
        # ====================================================

        # Aqui usamos a máscara de instância para descobrir
        # onde estão as fronteiras entre diferentes objetos.
        #
        # A função create_boundary_mask() foi criada
        # no arquivo preprocessing.py.
        boundary_mask = create_boundary_mask(
            instance_mask,
            boundary_width=self.boundary_width
        )


        # Estamos passando dois elementos:
        #
        # instance_mask
        #     -> informa quais pixels pertencem a cada objeto.
        #
        # boundary_width
        #     -> informa a largura da região de fronteira.
        #
        # Por exemplo, imagine:
        #
        #     objeto 1 | objeto 2
        #
        # A função pode identificar a região:
        #
        #     objeto 1 | FRONTEIRA | objeto 2
        #
        # Essa informação pode ser utilizada por uma rede
        # para aprender a separar objetos que estão próximos.


        # ====================================================
        # CONVERTENDO A BOUNDARY MASK PARA TENSOR
        # ====================================================

        # create_boundary_mask() aparentemente retorna
        # um array NumPy.
        #
        # Então precisamos convertê-lo para Tensor PyTorch.
        boundary_mask = torch.from_numpy(
            boundary_mask
        ).long()


        # Novamente utilizamos long porque a máscara
        # representa categorias/valores discretos.


        # ====================================================
        # RETORNO DA AMOSTRA
        # ====================================================

        # Retornamos todas as informações da amostra
        # organizadas em um dicionário.
        return {
            "image": image,

            "semantic_mask": semantic_mask,

            "instance_mask": instance_mask,

            "boundary_mask": boundary_mask,
        }