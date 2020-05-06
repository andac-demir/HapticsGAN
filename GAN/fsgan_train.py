import torch
from torch import nn
from torch.autograd import Variable
from fsgan_model import Encoder, Generator, Discriminator


class Train(object):
    def __init__(self, split, lr, batch_size, num_workers, epochs, optimization,
                 filename):
        self.encoder = torch.nn.DataParallel(Encoder().cuda())
        self.generator = torch.nn.DataParallel(Generator().cuda())
        self.discriminator = torch.nn.DataParallel(Discriminator().cuda())
        self.filename = filename

        # todo: custom weight initialization here

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs

        if optimization == 'adam':
            self.optim_g = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.lr,
                                            betas=(self.beta1, 0.999))
            self.optim_d = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.lr,
                                            betas=(self.beta1, 0.999))
        else:
            self.optim_g = torch.optim.LBFGS(self.generator.parameters(), lr=1,
                                             max_iter=20, max_eval=None,
                                             tolerance_grad=1e-05,
                                             tolerance_change=1e-09,
                                             history_size=100,
                                             line_search_fn=None)
            self.optim_d = torch.optim.LBFGS(self.discriminator.parameters(),
                                             lr=1, max_iter=20, max_eval=None,
                                             tolerance_grad=1e-05,
                                             tolerance_change=1e-09,
                                             history_size=100,
                                             line_search_fn=None)


    def save_model(self):
        '''
        Saves the model parameters of encoder, generator and discriminator
        under the directory: Model
        '''
        torch.save(self.encoder.state_dict(),
                   f="TrainedModels/encoder_%s.model" % self.filename)
        torch.save(self.generator.state_dict(),
                   f="TrainedModels/generator_%s.model" %self.filename)
        torch.save(self.discriminator.state_dict(),
                   f="TrainedModels/discriminator_%s.model" %self.filename)
        print("Models saved successfully.")

    def train_network(self):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0
        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                right_images = Variable(sample['right_images'].float()).cuda()
                right_embed = Variable(sample['right_embed'].float()).cuda()
                wrong_images = Variable(sample['wrong_images'].float()).cuda()

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))
                # Smoothing prevents the discriminator from overpowering the
                # generator adding penalty when the discriminator is
                # too confident - this avoids the generator lazily copying the
                # images in the training data.
                smoothed_real_labels = Variable(torch.FloatTensor(Utils.
                                                                  smooth_label(
                    real_labels.numpy(),
                    -0.1))).cuda()
                real_labels = Variable(real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images,
                                                              right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs
                outputs, _ = self.discriminator(wrong_images, right_embed)
                wrong_loss = criterion(outputs, fake_labels)

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs
                d_loss = real_loss + fake_loss + wrong_loss
                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images,
                                                              right_embed)
                _, activation_real = self.discriminator(right_images,
                                                        right_embed)
                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)
                # the first term in the loss function of the generator is the
                # regular cross entropy loss, the second term is
                # feature matching loss which measures the distance
                # between the real and generated images statistics
                # by comparing intermediate layers activations and
                # the third term is L1 distance between the generated and
                # real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly
                # to certain pixel values.
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake,
                                                  activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)
                g_loss.backward()
                self.optimG.step()
                if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch, d_loss, g_loss,
                                                  real_score, fake_score)
                    self.logger.draw(right_images, fake_images)

            self.logger.plot_epoch_w_scores(epoch)
            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator,
                                      self.checkpoints_path, epoch)

        # saves the trained model:
        self.save_model(self.encoder, self.generator, self.discriminator)
