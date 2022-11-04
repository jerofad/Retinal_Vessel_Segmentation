    # def test(self):
    #     self.model.eval()
    #     ious = torch.empty((0,))
    #     metrics_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     with torch.no_grad():
    #         for x, y in self.val_loader:
    #             image = x.to(self.device)
    #             mask = y.to(self.device)
    #             output = self.model(image)
    #             batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
    #             batch_score = [calculate_metrics(output_i, mask_j) for output_i, mask_j in zip(output, mask) ]
    #             metrics_scores = list(map(add, metrics_scores, batch_score))
    #             ious = torch.cat((ious, batch_ious.flatten()))
    #     return ious, metrics_scores


        # def validate_generalizability(self, epoch, plot=False):
    #     self.model.eval()
    #     ious = torch.empty((0,)).to(self.device)
    #     with torch.no_grad():
    #         for x, y, index in DataLoader(EtisDataset("Datasets/ETIS-LaribPolypDB")):
    #             image = x.to(self.device)
    #             mask = y.to(self.device)
    #             output = self.model(image)
    #             batch_ious = torch.mean(iou(output, mask))
    #             ious = torch.cat((ious, batch_ious.flatten()))
    #         return ious
