
      const uint i=thread_position_in_grid.x;
      const uint channel=i%C;
      const uint patch=(i/C)%4;
      const uint row=i/(4*C);
      const uint batch=row/S;
      const uint source=route[(row%S)*4+patch];
      out[i]=x[(batch*(S*4)+source)*C+channel];
