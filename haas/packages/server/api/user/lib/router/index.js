const express = require('express');
const controller = require('./controller');
const Auth = require('../auth');

// NOTE: difference between uM.verifyToken and uM.verifyTokenPermissive
// is the latter tries to refresh the token
class UserRouter{
  constructor(app, user) {
    const tokenManager = user.tokenManager;
    const UserModel = user.Model;
    const uM = user.middleware;

    const authStrategies = new Auth.AuthStrategies(app, user);

    const router = express.Router();
    const routeCtrl = controller(UserModel, tokenManager);

    router.use('/auth/', authStrategies.router);
    router.put('/', routeCtrl.create);

    //Every single route under user requires permission, besides create an auth

    const middleware = uM.verifyTokenPermissive({credentialsRequired: true});
    router.all('/*', middleware);

    router.get('/', uM.hasRole('admin'), routeCtrl.index);
    router.get('/public', routeCtrl.publicIndex);
    router.delete('/:id', uM.hasRole('admin'), routeCtrl.destroy);
    
    router.get('/me', routeCtrl.me);
    router.post('/me', routeCtrl.saveMe);
    router.delete('/me', routeCtrl.destroyMe);
    router.put('/:id/password', routeCtrl.changePassword);
    router.get('/:id', routeCtrl.show);
    

    this.router = router;
  }
}

module.exports = UserRouter;
